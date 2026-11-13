"""
zmq_test_pub.py — Testeur Publisher ZeroMQ
============================================
Fourni complet. Lancez ce script dans un terminal, puis zmq_test_sub.py
dans un autre pour vérifier la communication.

Usage :
    python zmq_test_pub.py
    python zmq_test_pub.py --port 5556 --topic test/hello
"""

import argparse
import time
import numpy as np
from zmq_utils import Publisher


def main(args):
    pub = Publisher(port=args.port, topic=args.topic)

    print(f"Publication toutes les {args.interval}s. Ctrl+C pour arrêter.")
    counter = 0

    try:
        while True:
            # On simule des données de détection
            fake_frame = np.random.randint(0, 255, (480, 848, 3), dtype=np.uint8)

            payload = {
                "timestamp": time.time(),
                "counter":   counter,
                "frame":     fake_frame,           # numpy array — sérialisé automatiquement
                "markers":   [
                    {"id": 0, "center": [320, 240], "angle": 0.5},
                    {"id": 1, "center": [100, 150], "angle": 1.2},
                ],
            }

            pub.send(payload)
            print(f"[PUB] Message #{counter} envoyé  (frame shape={fake_frame.shape})")
            counter += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[PUB] Arrêt.")
    finally:
        pub.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",     type=int,   default=5555)
    parser.add_argument("--topic",    type=str,   default="vision")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Intervalle entre deux messages (secondes)")
    main(parser.parse_args())
