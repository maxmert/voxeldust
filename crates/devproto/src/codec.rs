//! The JSON-lines codec — ONE encode/decode shared by the client's dev-control
//! listener and `vdctl`, so the two cannot drift to different JSON shapes. Branchless
//! shims: the single fallible edge is `decode_request`'s `map_err`; `encode_response`
//! is infallible because the `DevState` builder sanitizes every float to finite (a
//! non-finite float is the only thing that makes `serde_json` fail on these types).

use crate::dispatch::{DevError, DevRequest, DevResponse};

/// Decode one JSON line from vdctl into a [`DevRequest`].
///
/// # Errors
/// [`DevError::BadRequest`] if the line is not a valid `DevRequest`.
pub fn decode_request(line: &str) -> Result<DevRequest, DevError> {
    serde_json::from_str(line).map_err(|_| DevError::BadRequest)
}

/// Encode a [`DevResponse`] as one JSON line (no trailing newline; the caller frames).
#[must_use]
pub fn encode_response(resp: &DevResponse) -> String {
    serde_json::to_string(resp).expect("DevResponse floats are finite (builder-sanitized)")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::DevResponse;
    use crate::state::tests::sample;

    #[test]
    fn a_valid_line_decodes_a_bad_line_is_a_typed_error() {
        assert_eq!(
            decode_request("{\"cmd\":\"close\"}").expect("decode"),
            DevRequest::Close
        );
        assert_eq!(decode_request("not json"), Err(DevError::BadRequest));
        assert_eq!(
            decode_request("{\"cmd\":\"nonexistent\"}"),
            Err(DevError::BadRequest)
        );
    }

    #[test]
    fn responses_encode_to_one_decodable_line() {
        // Each response variant encodes and decodes back through the ONE codec.
        for resp in [
            DevResponse::Ack,
            DevResponse::State { state: sample() },
            DevResponse::Error {
                error: DevError::BadRequest,
            },
        ] {
            let line = encode_response(&resp);
            assert!(!line.contains('\n'), "one line, no embedded newline");
            assert_eq!(
                serde_json::from_str::<DevResponse>(&line).expect("decode"),
                resp
            );
        }
    }

    #[test]
    fn the_codec_is_the_single_shape_request_and_response_round_trip() {
        // vdctl encodes a request the client decodes; the client encodes a response
        // vdctl decodes — both through the SAME functions (no second shape).
        let req = DevRequest::Action {
            bit: 2,
            pressed: true,
        };
        let line = serde_json::to_string(&req).expect("vdctl encode");
        assert_eq!(decode_request(&line).expect("client decode"), req);
    }
}
