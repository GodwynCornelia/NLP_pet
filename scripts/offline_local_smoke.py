"""Real local inference in a container launched with --network none."""

import json
import socket

from support_router.common import setup_compute
from support_router.config import load_config


def main():
    def forbidden(*args, **kwargs):
        raise RuntimeError("offline_socket_forbidden")

    socket.create_connection = forbidden
    socket.socket.connect = forbidden
    config = load_config()
    _controller = setup_compute(config)
    from support_router.runtime import Router

    router = Router(config)
    prediction = router.predict("My card payment appears twice in the app.")
    assert prediction["predicted_category"] == "transaction_charged_twice"
    assert len(prediction["similar_examples"]) == 3
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "smoke",
                "run_id": router.run_id,
                "network": "none; all socket connects forbidden",
                "category": prediction["predicted_category"],
                "confidence": prediction["confidence"],
                "routing_decision": prediction["routing_decision"],
            }
        )
    )


if __name__ == "__main__":
    main()
