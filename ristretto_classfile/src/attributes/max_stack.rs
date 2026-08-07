use crate::attributes::{ExceptionTableEntry, Instruction};
use crate::verifiers::VerifyError;
use crate::{ConstantPool, Error, Result};
use ahash::{AHashMap, AHashSet};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct ReturnAddress {
    call_site: usize,
    target: usize,
    continuation: usize,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum StackValue {
    Category1,
    Category2,
    ReturnAddress(ReturnAddress),
    InvalidReturnAddress,
}

impl StackValue {
    const fn slots(self) -> u16 {
        match self {
            Self::Category2 => 2,
            Self::Category1 | Self::ReturnAddress(_) | Self::InvalidReturnAddress => 1,
        }
    }

    const fn is_category1(self) -> bool {
        !matches!(self, Self::Category2)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
struct StackId(usize);

impl StackId {
    const EMPTY: Self = Self(0);
}

#[derive(Clone, Copy, Debug)]
struct StackNode {
    value: StackValue,
    previous: StackId,
    depth: u16,
}

#[derive(Default)]
struct StackArena {
    nodes: Vec<StackNode>,
    interned: AHashMap<(StackValue, StackId), StackId>,
}

impl StackArena {
    fn depth(&self, stack: StackId) -> Result<u16> {
        if stack == StackId::EMPTY {
            return Ok(0);
        }

        let index = stack.0.checked_sub(1).ok_or_else(|| {
            verification_error("invalid operand-stack identifier".to_string())
        })?;
        self.nodes
            .get(index)
            .map(|node| node.depth)
            .ok_or_else(|| verification_error("invalid operand-stack identifier".to_string()))
    }

    fn push(&mut self, stack: StackId, value: StackValue) -> Result<StackId> {
        let key = (value, stack);
        if let Some(existing) = self.interned.get(&key) {
            return Ok(*existing);
        }

        let depth = self
            .depth(stack)?
            .checked_add(value.slots())
            .ok_or_else(|| verification_error("operand stack exceeds u16::MAX slots".to_string()))?;
        let raw_id = self
            .nodes
            .len()
            .checked_add(1)
            .ok_or_else(|| verification_error("too many operand-stack states".to_string()))?;
        let id = StackId(raw_id);
        self.nodes.push(StackNode {
            value,
            previous: stack,
            depth,
        });
        self.interned.insert(key, id);
        Ok(id)
    }

    fn pop(&self, stack: StackId) -> Result<(StackValue, StackId)> {
        let index = stack
            .0
            .checked_sub(1)
            .ok_or_else(|| verification_error("operand stack underflow".to_string()))?;
        let node = self
            .nodes
            .get(index)
            .ok_or_else(|| verification_error("invalid operand-stack identifier".to_string()))?;
        Ok((node.value, node.previous))
    }

    fn invalidate_return_addresses(
        &mut self,
        stack: StackId,
        inactive: &[ReturnAddress],
    ) -> Result<StackId> {
        let mut values = Vec::new();
        let mut cursor = stack;
        while cursor != StackId::EMPTY {
            let (mut value, previous) = self.pop(cursor)?;
            if let StackValue::ReturnAddress(address) = value
                && inactive.contains(&address)
            {
                value = StackValue::InvalidReturnAddress;
            }
            values.push(value);
            cursor = previous;
        }

        let mut rebuilt = StackId::EMPTY;
        for value in values.into_iter().rev() {
            rebuilt = self.push(rebuilt, value)?;
        }
        Ok(rebuilt)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum ReturnAddressLocal {
    Active(ReturnAddress),
    Invalid,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct FlowState {
    instruction: usize,
    stack: StackId,
    return_address_locals: Vec<(u16, ReturnAddressLocal)>,
    subroutines: Vec<ReturnAddress>,
}

#[derive(Default)]
struct Worklist {
    visited: AHashSet<FlowState>,
    queue: VecDeque<FlowState>,
    max_stack: u16,
}

impl Worklist {
    fn observe_stack(&mut self, stack: StackId, stacks: &StackArena) -> Result<()> {
        self.max_stack = self.max_stack.max(stacks.depth(stack)?);
        Ok(())
    }

    fn enqueue(&mut self, state: FlowState, stacks: &StackArena) -> Result<()> {
        self.observe_stack(state.stack, stacks)?;
        if self.visited.insert(state.clone()) {
            self.queue.push_back(state);
        }
        Ok(())
    }
}

fn verification_error(message: String) -> Error {
    Error::VerificationError(VerifyError::VerifyError(message))
}

fn checked_target(target: i64, instruction_count: usize) -> Result<usize> {
    let target = usize::try_from(target)?;
    if target >= instruction_count {
        return Err(Error::InvalidInstructionOffset(u32::try_from(target)?));
    }
    Ok(target)
}

fn relative_target(index: usize, relative: i32, instruction_count: usize) -> Result<usize> {
    let index = i64::try_from(index)?;
    checked_target(index + i64::from(relative), instruction_count)
}

fn next_instruction(index: usize, instruction_count: usize) -> Option<usize> {
    index
        .checked_add(1)
        .filter(|next| *next < instruction_count)
}

fn pop_value(stacks: &StackArena, stack: &mut StackId) -> Result<StackValue> {
    let (value, previous) = stacks.pop(*stack)?;
    *stack = previous;
    Ok(value)
}

fn pop_category1(stacks: &StackArena, stack: &mut StackId) -> Result<StackValue> {
    let value = pop_value(stacks, stack)?;
    if value.is_category1() {
        Ok(value)
    } else {
        Err(verification_error(
            "expected a category-1 value on the operand stack".to_string(),
        ))
    }
}

fn pop_slots(stacks: &StackArena, stack: &mut StackId, slots: u16) -> Result<()> {
    let mut remaining = slots;
    while remaining > 0 {
        let value = pop_value(stacks, stack)?;
        if matches!(
            value,
            StackValue::ReturnAddress(_) | StackValue::InvalidReturnAddress
        ) {
            return Err(verification_error(
                "returnAddress consumed by a non-returnAddress instruction".to_string(),
            ));
        }

        let value_slots = value.slots();
        if value_slots > remaining {
            return Err(verification_error(format!(
                "operand stack value occupies {value_slots} slots but only {remaining} slots are consumed"
            )));
        }
        remaining -= value_slots;
    }
    Ok(())
}

fn push_slots(
    stacks: &mut StackArena,
    stack: &mut StackId,
    slots: u16,
) -> Result<()> {
    let value = match slots {
        0 => return Ok(()),
        1 => StackValue::Category1,
        2 => StackValue::Category2,
        _ => {
            return Err(verification_error(format!(
                "instruction produces unsupported stack value width of {slots} slots"
            )));
        }
    };
    *stack = stacks.push(*stack, value)?;
    Ok(())
}

fn return_address_local(
    locals: &[(u16, ReturnAddressLocal)],
    index: u16,
) -> Option<ReturnAddressLocal> {
    locals
        .iter()
        .find_map(|(local_index, value)| (*local_index == index).then_some(*value))
}

fn clear_return_address_local(locals: &mut Vec<(u16, ReturnAddressLocal)>, index: u16) {
    locals.retain(|(local_index, _)| *local_index != index);
}

fn clear_return_address_local_range(
    locals: &mut Vec<(u16, ReturnAddressLocal)>,
    start: u16,
    slots: u16,
) {
    let end = u32::from(start) + u32::from(slots);
    locals.retain(|(local_index, _)| {
        let local_index = u32::from(*local_index);
        local_index < u32::from(start) || local_index >= end
    });
}

fn set_return_address_local(
    locals: &mut Vec<(u16, ReturnAddressLocal)>,
    index: u16,
    value: Option<ReturnAddressLocal>,
) {
    clear_return_address_local(locals, index);
    if let Some(value) = value {
        locals.push((index, value));
        locals.sort_unstable_by_key(|(local_index, _)| *local_index);
    }
}

fn invalidate_return_addresses(
    state: &mut FlowState,
    inactive: &[ReturnAddress],
    stacks: &mut StackArena,
) -> Result<()> {
    for (_, value) in &mut state.return_address_locals {
        if let ReturnAddressLocal::Active(address) = value
            && inactive.contains(address)
        {
            *value = ReturnAddressLocal::Invalid;
        }
    }

    state.stack = stacks.invalidate_return_addresses(state.stack, inactive)?;
    Ok(())
}

fn written_local(instruction: &Instruction) -> Option<(u16, u16)> {
    match instruction {
        Instruction::Istore(index)
        | Instruction::Fstore(index)
        | Instruction::Astore(index) => Some((u16::from(*index), 1)),
        Instruction::Lstore(index) | Instruction::Dstore(index) => Some((u16::from(*index), 2)),
        Instruction::Istore_0 | Instruction::Fstore_0 | Instruction::Astore_0 => Some((0, 1)),
        Instruction::Istore_1 | Instruction::Fstore_1 | Instruction::Astore_1 => Some((1, 1)),
        Instruction::Istore_2 | Instruction::Fstore_2 | Instruction::Astore_2 => Some((2, 1)),
        Instruction::Istore_3 | Instruction::Fstore_3 | Instruction::Astore_3 => Some((3, 1)),
        Instruction::Lstore_0 | Instruction::Dstore_0 => Some((0, 2)),
        Instruction::Lstore_1 | Instruction::Dstore_1 => Some((1, 2)),
        Instruction::Lstore_2 | Instruction::Dstore_2 => Some((2, 2)),
        Instruction::Lstore_3 | Instruction::Dstore_3 => Some((3, 2)),
        Instruction::Istore_w(index)
        | Instruction::Fstore_w(index)
        | Instruction::Astore_w(index) => Some((*index, 1)),
        Instruction::Lstore_w(index) | Instruction::Dstore_w(index) => Some((*index, 2)),
        _ => None,
    }
}

fn astore_index(instruction: &Instruction) -> Option<u16> {
    match instruction {
        Instruction::Astore(index) => Some(u16::from(*index)),
        Instruction::Astore_0 => Some(0),
        Instruction::Astore_1 => Some(1),
        Instruction::Astore_2 => Some(2),
        Instruction::Astore_3 => Some(3),
        Instruction::Astore_w(index) => Some(*index),
        _ => None,
    }
}

fn read_local(instruction: &Instruction) -> Option<(u16, u16)> {
    match instruction {
        Instruction::Iload(index)
        | Instruction::Fload(index)
        | Instruction::Aload(index)
        | Instruction::Iinc(index, _) => Some((u16::from(*index), 1)),
        Instruction::Lload(index) | Instruction::Dload(index) => Some((u16::from(*index), 2)),
        Instruction::Iload_0 | Instruction::Fload_0 | Instruction::Aload_0 => Some((0, 1)),
        Instruction::Iload_1 | Instruction::Fload_1 | Instruction::Aload_1 => Some((1, 1)),
        Instruction::Iload_2 | Instruction::Fload_2 | Instruction::Aload_2 => Some((2, 1)),
        Instruction::Iload_3 | Instruction::Fload_3 | Instruction::Aload_3 => Some((3, 1)),
        Instruction::Lload_0 | Instruction::Dload_0 => Some((0, 2)),
        Instruction::Lload_1 | Instruction::Dload_1 => Some((1, 2)),
        Instruction::Lload_2 | Instruction::Dload_2 => Some((2, 2)),
        Instruction::Lload_3 | Instruction::Dload_3 => Some((3, 2)),
        Instruction::Iload_w(index)
        | Instruction::Fload_w(index)
        | Instruction::Aload_w(index)
        | Instruction::Iinc_w(index, _) => Some((*index, 1)),
        Instruction::Lload_w(index) | Instruction::Dload_w(index) => Some((*index, 2)),
        _ => None,
    }
}

fn local_range_contains_return_address(
    locals: &[(u16, ReturnAddressLocal)],
    start: u16,
    slots: u16,
) -> bool {
    let start = u32::from(start);
    let end = start + u32::from(slots);
    locals.iter().any(|(local_index, _)| {
        let local_index = u32::from(*local_index);
        start <= local_index && local_index < end
    })
}

fn apply_astore(state: &mut FlowState, index: u16, stacks: &StackArena) -> Result<()> {
    let value = pop_category1(stacks, &mut state.stack)?;
    let return_address = match value {
        StackValue::ReturnAddress(address) => Some(ReturnAddressLocal::Active(address)),
        StackValue::InvalidReturnAddress => Some(ReturnAddressLocal::Invalid),
        StackValue::Category1 | StackValue::Category2 => None,
    };
    set_return_address_local(&mut state.return_address_locals, index, return_address);
    Ok(())
}

fn apply_pop(stack: &mut StackId, stacks: &StackArena) -> Result<()> {
    let _ = pop_category1(stacks, stack)?;
    Ok(())
}

fn apply_pop2(stack: &mut StackId, stacks: &StackArena) -> Result<()> {
    let first = pop_value(stacks, stack)?;
    if matches!(first, StackValue::Category2) {
        return Ok(());
    }

    let _ = pop_category1(stacks, stack)?;
    Ok(())
}

fn push_value(stacks: &mut StackArena, stack: &mut StackId, value: StackValue) -> Result<()> {
    *stack = stacks.push(*stack, value)?;
    Ok(())
}

fn apply_dup(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value)?;
    push_value(stacks, stack, value)
}

fn apply_dup_x1(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_category1(stacks, stack)?;
    let value2 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)
}

fn apply_dup_x2(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_category1(stacks, stack)?;
    let value2 = pop_value(stacks, stack)?;

    if matches!(value2, StackValue::Category2) {
        push_value(stacks, stack, value1)?;
        push_value(stacks, stack, value2)?;
        return push_value(stacks, stack, value1);
    }

    let value3 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value3)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)
}

fn apply_dup2(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_value(stacks, stack)?;
    if matches!(value1, StackValue::Category2) {
        push_value(stacks, stack, value1)?;
        return push_value(stacks, stack, value1);
    }

    let value2 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)
}

fn apply_dup2_x1(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_value(stacks, stack)?;
    if matches!(value1, StackValue::Category2) {
        let value2 = pop_category1(stacks, stack)?;
        push_value(stacks, stack, value1)?;
        push_value(stacks, stack, value2)?;
        return push_value(stacks, stack, value1);
    }

    let value2 = pop_category1(stacks, stack)?;
    let value3 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value3)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)
}

fn apply_dup2_x2(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_value(stacks, stack)?;

    if matches!(value1, StackValue::Category2) {
        let value2 = pop_value(stacks, stack)?;
        if matches!(value2, StackValue::Category2) {
            push_value(stacks, stack, value1)?;
            push_value(stacks, stack, value2)?;
            return push_value(stacks, stack, value1);
        }

        let value3 = pop_category1(stacks, stack)?;
        push_value(stacks, stack, value1)?;
        push_value(stacks, stack, value3)?;
        push_value(stacks, stack, value2)?;
        return push_value(stacks, stack, value1);
    }

    let value2 = pop_category1(stacks, stack)?;
    let value3 = pop_value(stacks, stack)?;

    if matches!(value3, StackValue::Category2) {
        push_value(stacks, stack, value2)?;
        push_value(stacks, stack, value1)?;
        push_value(stacks, stack, value3)?;
        push_value(stacks, stack, value2)?;
        return push_value(stacks, stack, value1);
    }

    let value4 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value4)?;
    push_value(stacks, stack, value3)?;
    push_value(stacks, stack, value2)?;
    push_value(stacks, stack, value1)
}

fn apply_swap(stack: &mut StackId, stacks: &mut StackArena) -> Result<()> {
    let value1 = pop_category1(stacks, stack)?;
    let value2 = pop_category1(stacks, stack)?;
    push_value(stacks, stack, value1)?;
    push_value(stacks, stack, value2)
}

fn apply_stack_manipulation(
    instruction: &Instruction,
    stack: &mut StackId,
    stacks: &mut StackArena,
) -> Result<bool> {
    match instruction {
        Instruction::Pop => apply_pop(stack, stacks)?,
        Instruction::Pop2 => apply_pop2(stack, stacks)?,
        Instruction::Dup => apply_dup(stack, stacks)?,
        Instruction::Dup_x1 => apply_dup_x1(stack, stacks)?,
        Instruction::Dup_x2 => apply_dup_x2(stack, stacks)?,
        Instruction::Dup2 => apply_dup2(stack, stacks)?,
        Instruction::Dup2_x1 => apply_dup2_x1(stack, stacks)?,
        Instruction::Dup2_x2 => apply_dup2_x2(stack, stacks)?,
        Instruction::Swap => apply_swap(stack, stacks)?,
        _ => return Ok(false),
    }
    Ok(true)
}

fn apply_instruction(
    state: &mut FlowState,
    instruction: &Instruction,
    constant_pool: &ConstantPool<'_>,
    stacks: &mut StackArena,
) -> Result<()> {
    if let Some(index) = astore_index(instruction) {
        return apply_astore(state, index, stacks);
    }

    if apply_stack_manipulation(instruction, &mut state.stack, stacks)? {
        return Ok(());
    }

    if let Some((index, slots)) = read_local(instruction)
        && local_range_contains_return_address(&state.return_address_locals, index, slots)
    {
        return Err(verification_error(format!(
            "instruction reads returnAddress local variable starting at {index}"
        )));
    }

    let (popped_slots, pushed_slots) = instruction.stack_effect(constant_pool)?;
    pop_slots(stacks, &mut state.stack, popped_slots)?;
    push_slots(stacks, &mut state.stack, pushed_slots)?;

    if let Some((index, slots)) = written_local(instruction) {
        clear_return_address_local_range(&mut state.return_address_locals, index, slots);
    }

    Ok(())
}

fn validate_exception_table(
    exception_table: &[ExceptionTableEntry],
    instruction_count: usize,
) -> Result<()> {
    for entry in exception_table {
        let start = usize::from(entry.range_pc.start);
        let end = usize::from(entry.range_pc.end);
        let handler = usize::from(entry.handler_pc);

        if start >= end {
            return Err(Error::InvalidInstructionOffset(u32::try_from(start)?));
        }
        if end > instruction_count {
            return Err(Error::InvalidInstructionOffset(u32::try_from(end)?));
        }
        if handler >= instruction_count {
            return Err(Error::InvalidInstructionOffset(u32::try_from(handler)?));
        }
    }
    Ok(())
}

fn enqueue_exception_handlers(
    worklist: &mut Worklist,
    stacks: &mut StackArena,
    state: &FlowState,
    exception_table: &[ExceptionTableEntry],
) -> Result<()> {
    for entry in exception_table {
        let start = usize::from(entry.range_pc.start);
        let end = usize::from(entry.range_pc.end);
        if start <= state.instruction && state.instruction < end {
            let mut handler_state = state.clone();
            handler_state.instruction = usize::from(entry.handler_pc);
            handler_state.stack = stacks.push(StackId::EMPTY, StackValue::Category1)?;
            worklist.enqueue(handler_state, stacks)?;
        }
    }
    Ok(())
}

fn enqueue_normal_successors(
    worklist: &mut Worklist,
    stacks: &StackArena,
    state: &FlowState,
    instruction: &Instruction,
    instruction_count: usize,
) -> Result<()> {
    let mut enqueue_target = |target: usize| -> Result<()> {
        let mut successor = state.clone();
        successor.instruction = target;
        worklist.enqueue(successor, stacks)
    };

    match instruction {
        Instruction::Ifeq(target)
        | Instruction::Ifne(target)
        | Instruction::Iflt(target)
        | Instruction::Ifge(target)
        | Instruction::Ifgt(target)
        | Instruction::Ifle(target)
        | Instruction::If_icmpeq(target)
        | Instruction::If_icmpne(target)
        | Instruction::If_icmplt(target)
        | Instruction::If_icmpge(target)
        | Instruction::If_icmpgt(target)
        | Instruction::If_icmple(target)
        | Instruction::If_acmpeq(target)
        | Instruction::If_acmpne(target)
        | Instruction::Ifnull(target)
        | Instruction::Ifnonnull(target) => {
            enqueue_target(checked_target(i64::from(*target), instruction_count)?)?;
            if let Some(next) = next_instruction(state.instruction, instruction_count) {
                enqueue_target(next)?;
            }
        }
        Instruction::Goto(target) => {
            enqueue_target(checked_target(i64::from(*target), instruction_count)?)?;
        }
        Instruction::Goto_w(target) => {
            enqueue_target(checked_target(i64::from(*target), instruction_count)?)?;
        }
        Instruction::Tableswitch(table) => {
            enqueue_target(relative_target(
                state.instruction,
                table.default,
                instruction_count,
            )?)?;
            for relative in &table.offsets {
                enqueue_target(relative_target(
                    state.instruction,
                    *relative,
                    instruction_count,
                )?)?;
            }
        }
        Instruction::Lookupswitch(lookup) => {
            enqueue_target(relative_target(
                state.instruction,
                lookup.default,
                instruction_count,
            )?)?;
            for relative in lookup.pairs.values() {
                enqueue_target(relative_target(
                    state.instruction,
                    *relative,
                    instruction_count,
                )?)?;
            }
        }
        Instruction::Ireturn
        | Instruction::Lreturn
        | Instruction::Freturn
        | Instruction::Dreturn
        | Instruction::Areturn
        | Instruction::Return
        | Instruction::Athrow
        | Instruction::Ret(..)
        | Instruction::Ret_w(..)
        | Instruction::Jsr(..)
        | Instruction::Jsr_w(..) => {}
        _ => {
            if let Some(next) = next_instruction(state.instruction, instruction_count) {
                enqueue_target(next)?;
            }
        }
    }
    Ok(())
}

fn enqueue_jsr(
    worklist: &mut Worklist,
    stacks: &mut StackArena,
    mut state: FlowState,
    target: usize,
    instruction_count: usize,
) -> Result<()> {
    let continuation = next_instruction(state.instruction, instruction_count).ok_or_else(|| {
        verification_error(format!(
            "jsr at instruction {} has no continuation",
            state.instruction
        ))
    })?;

    if state.subroutines.iter().any(|frame| frame.target == target) {
        return Err(verification_error(format!(
            "recursive jsr subroutine call to instruction {target}"
        )));
    }

    let address = ReturnAddress {
        call_site: state.instruction,
        target,
        continuation,
    };
    state.stack = stacks.push(state.stack, StackValue::ReturnAddress(address))?;
    state.subroutines.push(address);
    state.instruction = target;
    worklist.enqueue(state, stacks)
}

fn enqueue_ret(
    worklist: &mut Worklist,
    stacks: &mut StackArena,
    mut state: FlowState,
    local_index: u16,
) -> Result<()> {
    let address = match return_address_local(&state.return_address_locals, local_index) {
        Some(ReturnAddressLocal::Active(address)) => address,
        Some(ReturnAddressLocal::Invalid) => {
            return Err(verification_error(format!(
                "ret at instruction {} reuses an inactive returnAddress from local {local_index}",
                state.instruction
            )));
        }
        None => {
            return Err(verification_error(format!(
                "ret at instruction {} reads local {local_index} without a returnAddress",
                state.instruction
            )));
        }
    };

    let frame_index = state
        .subroutines
        .iter()
        .position(|frame| *frame == address)
        .ok_or_else(|| {
            verification_error(format!(
                "ret at instruction {} targets an inactive jsr return address",
                state.instruction
            ))
        })?;

    let inactive: Vec<ReturnAddress> = state.subroutines.drain(frame_index..).collect();
    invalidate_return_addresses(&mut state, &inactive, stacks)?;
    state.instruction = address.continuation;
    worklist.enqueue(state, stacks)
}

fn analyze(
    instructions: &[Instruction],
    constant_pool: &ConstantPool<'_>,
    exception_table: &[ExceptionTableEntry],
) -> Result<u16> {
    if instructions.is_empty() {
        if exception_table.is_empty() {
            return Ok(0);
        }
        return Err(Error::InvalidInstructionOffset(0));
    }

    validate_exception_table(exception_table, instructions.len())?;

    let mut stacks = StackArena::default();
    let mut worklist = Worklist::default();
    worklist.enqueue(
        FlowState {
            instruction: 0,
            stack: StackId::EMPTY,
            return_address_locals: Vec::new(),
            subroutines: Vec::new(),
        },
        &stacks,
    )?;

    while let Some(mut state) = worklist.queue.pop_front() {
        let instruction = instructions
            .get(state.instruction)
            .ok_or(Error::InvalidInstructionOffset(u32::try_from(
                state.instruction,
            )?))?;

        enqueue_exception_handlers(&mut worklist, &mut stacks, &state, exception_table)?;

        match instruction {
            Instruction::Jsr(target) => {
                let target = checked_target(i64::from(*target), instructions.len())?;
                enqueue_jsr(
                    &mut worklist,
                    &mut stacks,
                    state,
                    target,
                    instructions.len(),
                )?;
            }
            Instruction::Jsr_w(target) => {
                let target = checked_target(i64::from(*target), instructions.len())?;
                enqueue_jsr(
                    &mut worklist,
                    &mut stacks,
                    state,
                    target,
                    instructions.len(),
                )?;
            }
            Instruction::Ret(index) => {
                enqueue_ret(&mut worklist, &mut stacks, state, u16::from(*index))?;
            }
            Instruction::Ret_w(index) => {
                enqueue_ret(&mut worklist, &mut stacks, state, *index)?;
            }
            _ => {
                apply_instruction(&mut state, instruction, constant_pool, &mut stacks)?;
                worklist.observe_stack(state.stack, &stacks)?;
                enqueue_normal_successors(
                    &mut worklist,
                    &stacks,
                    &state,
                    instruction,
                    instructions.len(),
                )?;
            }
        }
    }

    Ok(worklist.max_stack)
}

/// Calculates the maximum operand-stack depth for an instruction sequence, including exception
/// handlers.
///
/// The result is expressed in JVM stack slots and is suitable for the `max_stack` field of a
/// method's `Code` attribute. The analysis follows all branch and switch targets, exception-handler
/// edges, and legacy `jsr`/`ret` subroutine return addresses.
///
/// # Errors
///
/// Returns an error if an instruction target, exception-table entry, field descriptor, method
/// descriptor, return address, or operand-stack state is invalid.
///
/// # References
///
/// - [JVMS §2.6.2](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-2.html#jvms-2.6.2)
/// - [JVMS §4.7.3](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html#jvms-4.7.3)
/// - [JVMS §4.10.2.5](https://docs.oracle.com/javase/specs/jvms/se7/html/jvms-4.html#jvms-4.10.2.5)
pub fn max_stack_with_exception_table(
    instructions: &[Instruction],
    constant_pool: &ConstantPool<'_>,
    exception_table: &[ExceptionTableEntry],
) -> Result<u16> {
    analyze(instructions, constant_pool, exception_table)
}

/// Trait for calculating the maximum operand-stack size required by JVM bytecode instructions.
///
/// # Examples
///
/// ```rust
/// use ristretto_classfile::attributes::{Instruction, MaxStack};
/// use ristretto_classfile::ConstantPool;
///
/// let constant_pool = ConstantPool::new();
/// let instructions = [
///     Instruction::Iconst_0,
///     Instruction::Iconst_1,
///     Instruction::Pop,
///     Instruction::Pop,
///     Instruction::Return,
/// ];
///
/// let max_size = instructions.max_stack(&constant_pool)?;
/// assert_eq!(max_size, 2);
/// # Ok::<(), ristretto_classfile::Error>(())
/// ```
///
/// # References
///
/// - [JVMS §2.6.2](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-2.html#jvms-2.6.2)
/// - [JVMS §4.7.3](https://docs.oracle.com/javase/specs/jvms/se25/html/jvms-4.html#jvms-4.7.3)
pub trait MaxStack {
    /// Calculates the maximum operand-stack depth reachable from the method entry.
    ///
    /// The result is expressed in JVM stack slots. Use [`max_stack_with_exception_table`] when the
    /// method has exception handlers.
    ///
    /// # Errors
    ///
    /// Returns an error if an instruction target, field descriptor, method descriptor, return
    /// address, or operand-stack state is invalid.
    fn max_stack(&self, constant_pool: &ConstantPool<'_>) -> Result<u16>;
}

impl MaxStack for [Instruction] {
    fn max_stack(&self, constant_pool: &ConstantPool<'_>) -> Result<u16> {
        analyze(self, constant_pool, &[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attributes::{LookupSwitch, TableSwitch};
    use indexmap::IndexMap;

    #[test]
    #[expect(clippy::useless_vec)]
    fn test_max_stack_vec() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = vec![Instruction::Iconst_0, Instruction::Return];
        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_empty() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [];
        assert_eq!(instructions.max_stack(&constant_pool)?, 0);
        Ok(())
    }

    #[test]
    fn test_max_stack_return() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [Instruction::Return];
        assert_eq!(instructions.max_stack(&constant_pool)?, 0);
        Ok(())
    }

    #[test]
    fn test_max_stack_two_constants() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Iconst_1,
            Instruction::Pop,
            Instruction::Pop,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_pop_single_constant() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Pop,
            Instruction::Iconst_1,
            Instruction::Pop,
            Instruction::Iconst_2,
            Instruction::Pop,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_category_two_values() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Lconst_0,
            Instruction::Dconst_0,
            Instruction::Pop2,
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 4);
        Ok(())
    }

    #[test]
    fn test_max_stack_category_two_conversion_and_negation() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::I2l,
            Instruction::Lneg,
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_category_two_fields() -> Result<()> {
        let mut constant_pool = ConstantPool::new();
        let class_index = constant_pool.add_class("Foo")?;
        let field_index = constant_pool.add_field_ref(class_index, "value", "J")?;

        let getstatic = [
            Instruction::Getstatic(field_index),
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(getstatic.max_stack(&constant_pool)?, 2);

        let getfield = [
            Instruction::Aconst_null,
            Instruction::Getfield(field_index),
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(getfield.max_stack(&constant_pool)?, 2);

        let putfield = [
            Instruction::Aconst_null,
            Instruction::Lconst_0,
            Instruction::Putfield(field_index),
            Instruction::Return,
        ];
        assert_eq!(putfield.max_stack(&constant_pool)?, 3);

        Ok(())
    }

    #[test]
    fn test_max_stack_branches_do_not_accumulate_both_paths() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Ifeq(4),
            Instruction::Lconst_0,
            Instruction::Goto(5),
            Instruction::Lconst_0,
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_switch_uses_all_targets() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Tableswitch(Box::new(TableSwitch {
                default: 1,
                low: 0,
                high: 1,
                offsets: vec![1, 3],
            })),
            Instruction::Lconst_0,
            Instruction::Goto(6),
            Instruction::Dconst_0,
            Instruction::Goto(6),
            Instruction::Pop2,
            Instruction::Return,
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_lookup_switch_uses_all_targets() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Lookupswitch(Box::new(LookupSwitch {
                default: 1,
                pairs: IndexMap::from([(1, 1), (2, 3)]),
            })),
            Instruction::Lconst_0,
            Instruction::Goto(6),
            Instruction::Dconst_0,
            Instruction::Goto(6),
            Instruction::Pop2,
            Instruction::Return,
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_loop_reaches_fixed_point() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Istore_0,
            Instruction::Iload_0,
            Instruction::Ifeq(6),
            Instruction::Iinc(0, 1),
            Instruction::Goto(2),
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_exception_handler_entry() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Aconst_null,
            Instruction::Athrow,
            Instruction::Astore_0,
            Instruction::Lconst_0,
            Instruction::Pop2,
            Instruction::Return,
        ];
        let exception_table = [ExceptionTableEntry {
            range_pc: 0..2,
            handler_pc: 2,
            catch_type: 0,
        }];

        assert_eq!(
            max_stack_with_exception_table(&instructions, &constant_pool, &exception_table)?,
            2
        );
        Ok(())
    }

    #[test]
    fn test_max_stack_jsr_ret_uses_actual_return_depth() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Iconst_0,
            Instruction::Jsr(5),
            Instruction::Iadd,
            Instruction::Pop,
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Iconst_1,
            Instruction::Ret(0),
        ];

        // The subroutine returns with the original int plus a second int on the operand stack.
        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_jsr_return_address_survives_dup_and_swap() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Dup,
            Instruction::Aconst_null,
            Instruction::Swap,
            Instruction::Astore_0,
            Instruction::Pop,
            Instruction::Pop,
            Instruction::Ret(0),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 3);
        Ok(())
    }

    #[test]
    fn test_max_stack_nested_jsr_ret() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Lconst_0,
            Instruction::Pop2,
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Jsr(8),
            Instruction::Ret(0),
            Instruction::Return,
            Instruction::Astore_1,
            Instruction::Dconst_0,
            Instruction::Pop2,
            Instruction::Ret(1),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_ret_can_exit_outer_subroutine() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Lconst_0,
            Instruction::Pop2,
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Jsr(8),
            Instruction::Ret(0),
            Instruction::Return,
            Instruction::Astore_1,
            Instruction::Ret(0),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 2);
        Ok(())
    }

    #[test]
    fn test_max_stack_exception_handler_preserves_return_address_locals() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Return,
            Instruction::Return,
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Aconst_null,
            Instruction::Athrow,
            Instruction::Astore_1,
            Instruction::Ret(0),
        ];
        let exception_table = [ExceptionTableEntry {
            range_pc: 5..7,
            handler_pc: 7,
            catch_type: 0,
        }];

        assert_eq!(
            max_stack_with_exception_table(&instructions, &constant_pool, &exception_table)?,
            1
        );
        Ok(())
    }

    #[test]
    fn test_max_stack_category_two_stack_manipulation() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Lconst_0,
            Instruction::Dup2,
            Instruction::Pop2,
            Instruction::Pop2,
            Instruction::Return,
        ];
        assert_eq!(instructions.max_stack(&constant_pool)?, 4);
        Ok(())
    }

    #[test]
    fn test_max_stack_reenters_same_jsr_call_site_after_ret() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Iconst_0,
            Instruction::Ifeq(0),
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Ret(0),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_same_subroutine_from_multiple_call_sites() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Jsr(4),
            Instruction::Return,
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Ret(0),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_wide_jsr_ret() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr_w(3),
            Instruction::Return,
            Instruction::Return,
            Instruction::Astore_w(300),
            Instruction::Ret_w(300),
        ];

        assert_eq!(instructions.max_stack(&constant_pool)?, 1);
        Ok(())
    }

    #[test]
    fn test_max_stack_rejects_recursive_jsr() {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(2),
            Instruction::Return,
            Instruction::Astore_0,
            Instruction::Jsr(2),
            Instruction::Ret(0),
        ];

        assert!(matches!(
            instructions.max_stack(&constant_pool),
            Err(Error::VerificationError(_))
        ));
    }

    #[test]
    fn test_max_stack_rejects_ret_without_return_address() {
        let constant_pool = ConstantPool::new();
        let instructions = [Instruction::Ret(0)];

        assert!(matches!(
            instructions.max_stack(&constant_pool),
            Err(Error::VerificationError(_))
        ));
    }

    #[test]
    fn test_max_stack_rejects_reused_return_address() {
        let constant_pool = ConstantPool::new();
        let instructions = [
            Instruction::Jsr(4),
            Instruction::Astore_1,
            Instruction::Ret(1),
            Instruction::Return,
            Instruction::Dup,
            Instruction::Astore_0,
            Instruction::Ret(0),
        ];

        assert!(matches!(
            instructions.max_stack(&constant_pool),
            Err(Error::VerificationError(_))
        ));
    }

    #[test]
    #[expect(clippy::too_many_lines)]
    fn test_max_stack_all_legal_dup_forms() -> Result<()> {
        let constant_pool = ConstantPool::new();
        let cases = [
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Iconst_2,
                    Instruction::Dup_x2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                4,
            ),
            (
                vec![
                    Instruction::Lconst_0,
                    Instruction::Iconst_0,
                    Instruction::Dup_x2,
                    Instruction::Pop,
                    Instruction::Pop2,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                4,
            ),
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Dup2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                4,
            ),
            (
                vec![
                    Instruction::Lconst_0,
                    Instruction::Dup2,
                    Instruction::Pop2,
                    Instruction::Pop2,
                    Instruction::Return,
                ],
                4,
            ),
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Iconst_2,
                    Instruction::Dup2_x1,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                5,
            ),
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Lconst_0,
                    Instruction::Dup2_x1,
                    Instruction::Pop2,
                    Instruction::Pop,
                    Instruction::Pop2,
                    Instruction::Return,
                ],
                5,
            ),
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Iconst_2,
                    Instruction::Iconst_3,
                    Instruction::Dup2_x2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                6,
            ),
            (
                vec![
                    Instruction::Lconst_0,
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Dup2_x2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Return,
                ],
                6,
            ),
            (
                vec![
                    Instruction::Iconst_0,
                    Instruction::Iconst_1,
                    Instruction::Lconst_0,
                    Instruction::Dup2_x2,
                    Instruction::Pop2,
                    Instruction::Pop,
                    Instruction::Pop,
                    Instruction::Pop2,
                    Instruction::Return,
                ],
                6,
            ),
            (
                vec![
                    Instruction::Lconst_0,
                    Instruction::Lconst_1,
                    Instruction::Dup2_x2,
                    Instruction::Pop2,
                    Instruction::Pop2,
                    Instruction::Pop2,
                    Instruction::Return,
                ],
                6,
            ),
        ];

        for (instructions, expected) in cases {
            assert_eq!(instructions.max_stack(&constant_pool)?, expected);
        }
        Ok(())
    }

    #[test]
    fn test_max_stack_rejects_invalid_stack_manipulation_forms() {
        let constant_pool = ConstantPool::new();
        let cases = [
            vec![Instruction::Lconst_0, Instruction::Dup, Instruction::Return],
            vec![
                Instruction::Iconst_0,
                Instruction::Lconst_0,
                Instruction::Swap,
                Instruction::Return,
            ],
            vec![
                Instruction::Lconst_0,
                Instruction::Iconst_0,
                Instruction::Dup2,
                Instruction::Return,
            ],
        ];

        for instructions in cases {
            assert!(matches!(
                instructions.max_stack(&constant_pool),
                Err(Error::VerificationError(_))
            ));
        }
    }

    #[test]
    fn test_max_stack_rejects_stack_underflow() {
        let constant_pool = ConstantPool::new();
        let instructions = [Instruction::Pop, Instruction::Return];

        assert!(matches!(
            instructions.max_stack(&constant_pool),
            Err(Error::VerificationError(_))
        ));
    }
}
