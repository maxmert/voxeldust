//! Signal converter — condition → action rules for transforming signals.

use super::types::SignalValue;

/// A single condition → action rule in a Signal Converter block.
#[derive(Clone, Debug)]
pub struct SignalRule {
    /// Channel to read input from.
    pub input_channel: String,
    /// Condition to evaluate on the input value.
    pub condition: SignalCondition,
    /// Channel to write output to.
    pub output_channel: String,
    /// Expression to compute the output value.
    pub expression: SignalExpression,
}

/// Condition to evaluate on an input signal value.
#[derive(Clone, Debug, PartialEq)]
pub enum SignalCondition {
    /// Input float > threshold.
    GreaterThan(f32),
    /// Input float < threshold.
    LessThan(f32),
    /// Input float ≈ value (within epsilon 0.001).
    Equals(f32),
    /// Input float ≠ value.
    NotEquals(f32),
    /// Input changed from previous tick (dirty flag on the channel).
    Changed,
    /// Always fires every tick.
    Always,
}

impl SignalCondition {
    /// Evaluate this condition against an input signal value.
    /// `channel_dirty` indicates whether the input channel changed this tick.
    pub fn evaluate(&self, input: SignalValue, channel_dirty: bool) -> bool {
        let f = input.as_f32();
        match self {
            Self::GreaterThan(t) => f > *t,
            Self::LessThan(t) => f < *t,
            Self::Equals(v) => (f - v).abs() < 0.001,
            Self::NotEquals(v) => (f - v).abs() >= 0.001,
            Self::Changed => channel_dirty,
            Self::Always => true,
        }
    }
}

/// Expression to compute an output signal value from an input.
#[derive(Clone, Debug, PartialEq)]
pub enum SignalExpression {
    /// Output a fixed constant value.
    Constant(SignalValue),
    /// Pass the input value through unchanged.
    PassThrough,
    /// Invert: !bool or 1.0 - float.
    Invert,
    /// Multiply input by a scale factor.
    Scale(f32),
    /// Clamp input to [min, max] range.
    Clamp(f32, f32),
}

impl SignalExpression {
    /// Compute the output value from an input.
    pub fn compute(&self, input: SignalValue) -> SignalValue {
        match self {
            Self::Constant(v) => *v,
            Self::PassThrough => input,
            Self::Invert => {
                match input {
                    SignalValue::Bool(b) => SignalValue::Bool(!b),
                    SignalValue::Float(f) => SignalValue::Float(1.0 - f),
                    SignalValue::State(s) => SignalValue::State(if s == 0 { 1 } else { 0 }),
                }
            }
            Self::Scale(factor) => {
                SignalValue::Float(input.as_f32() * factor)
            }
            Self::Clamp(min, max) => {
                SignalValue::Float(input.as_f32().clamp(*min, *max))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_greater_than() {
        assert!(SignalCondition::GreaterThan(50.0).evaluate(SignalValue::Float(60.0), false));
        assert!(!SignalCondition::GreaterThan(50.0).evaluate(SignalValue::Float(40.0), false));
    }

    #[test]
    fn condition_less_than() {
        assert!(SignalCondition::LessThan(50.0).evaluate(SignalValue::Float(30.0), false));
        assert!(!SignalCondition::LessThan(50.0).evaluate(SignalValue::Float(60.0), false));
    }

    #[test]
    fn condition_equals() {
        assert!(SignalCondition::Equals(5.0).evaluate(SignalValue::Float(5.0), false));
        assert!(!SignalCondition::Equals(5.0).evaluate(SignalValue::Float(5.01), false));
    }

    #[test]
    fn condition_changed() {
        assert!(SignalCondition::Changed.evaluate(SignalValue::Float(0.0), true));
        assert!(!SignalCondition::Changed.evaluate(SignalValue::Float(0.0), false));
    }

    #[test]
    fn condition_always() {
        assert!(SignalCondition::Always.evaluate(SignalValue::Float(0.0), false));
    }

    #[test]
    fn expression_constant() {
        let expr = SignalExpression::Constant(SignalValue::Bool(true));
        assert_eq!(expr.compute(SignalValue::Float(0.0)), SignalValue::Bool(true));
    }

    #[test]
    fn expression_passthrough() {
        let input = SignalValue::Float(42.0);
        assert_eq!(SignalExpression::PassThrough.compute(input), input);
    }

    #[test]
    fn expression_invert() {
        assert_eq!(
            SignalExpression::Invert.compute(SignalValue::Bool(true)),
            SignalValue::Bool(false)
        );
        assert_eq!(
            SignalExpression::Invert.compute(SignalValue::Float(0.3)),
            SignalValue::Float(0.7)
        );
    }

    #[test]
    fn expression_scale() {
        assert_eq!(
            SignalExpression::Scale(2.0).compute(SignalValue::Float(5.0)),
            SignalValue::Float(10.0)
        );
    }

    #[test]
    fn expression_clamp() {
        assert_eq!(
            SignalExpression::Clamp(0.0, 1.0).compute(SignalValue::Float(1.5)),
            SignalValue::Float(1.0)
        );
        assert_eq!(
            SignalExpression::Clamp(0.0, 1.0).compute(SignalValue::Float(-0.5)),
            SignalValue::Float(0.0)
        );
    }

    #[test]
    fn full_rule_evaluation() {
        let rule = SignalRule {
            input_channel: "pressure".into(),
            condition: SignalCondition::LessThan(50.0),
            output_channel: "alarm".into(),
            expression: SignalExpression::Constant(SignalValue::Bool(true)),
        };

        let input = SignalValue::Float(30.0);
        assert!(rule.condition.evaluate(input, true));
        assert_eq!(rule.expression.compute(input), SignalValue::Bool(true));
    }
}
