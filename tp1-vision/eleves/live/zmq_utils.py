"""
zmq_utils.py — Utilitaires ZeroMQ pour la communication inter-processus
==========================================================================
Fourni complet. Vous n'avez pas à modifier ce fichier.

Pattern utilisé : PUB/SUB
  - Un Publisher envoie des données (frame, détections, etc.)
  - Un ou plusieurs Subscribers reçoivent ces données
  - Le découplage permet à un programme C++ d'asservissement de s'abonner
    sans modifier ce code.

Sérialisation : msgpack + gestion spéciale pour les tableaux numpy (ndarray).
"""

import zmq
import msgpack
import numpy as np
import time


# ---------------------------------------------------------------------------
# Encodage / décodage des numpy arrays dans msgpack
# ---------------------------------------------------------------------------

def _encode_ndarray(obj):
    """Sérialise un ndarray numpy en dict msgpack-compatible."""
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"dtype": str(obj.dtype),
            b"shape": list(obj.shape),
            b"data": obj.tobytes(),
        }
    raise TypeError(f"Type non sérialisable : {type(obj)}")


def _decode_ndarray(obj):
    """Désérialise un dict msgpack en ndarray numpy si applicable."""
    if b"__ndarray__" in obj:
        return np.frombuffer(
            obj[b"data"],
            dtype=np.dtype(obj[b"dtype"])
        ).reshape(obj[b"shape"])
    return obj


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------

class Publisher:
    """
    Publie des données structurées sur un port TCP.

    Exemple :
        pub = Publisher(port=5555, topic="vision/perception")
        pub.send({"timestamp": time.time(), "markers": [...]})
    """

    def __init__(self, port: int = 5555, topic: str = "vision"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        self.topic = topic.encode()
        # Petite pause pour laisser le socket s'initialiser
        time.sleep(0.1)
        print(f"[Publisher] En écoute sur le port {port}, topic='{topic}'")

    def send(self, data: dict):
        """Sérialise et envoie un dictionnaire Python."""
        payload = msgpack.packb(data, default=_encode_ndarray, use_bin_type=True)
        self.sock.send_multipart([self.topic, payload])

    def close(self):
        self.sock.close()
        self.ctx.term()


# ---------------------------------------------------------------------------
# Subscriber
# ---------------------------------------------------------------------------

class Subscriber:
    """
    S'abonne à un topic ZeroMQ et reçoit des données.

    Exemple :
        sub = Subscriber(host="localhost", port=5555, topic="vision/perception")
        while True:
            data = sub.receive(timeout_ms=100)
            if data:
                print(data["timestamp"])
    """

    def __init__(self, host: str = "localhost", port: int = 5555, topic: str = "vision"):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.connect(f"tcp://{host}:{port}")
        self.sock.setsockopt(zmq.SUBSCRIBE, topic.encode())
        print(f"[Subscriber] Connecté à {host}:{port}, topic='{topic}'")

    def receive(self, timeout_ms: int = 0) -> dict | None:
        """
        Reçoit un message.

        Args:
            timeout_ms : attente max en ms. 0 = bloquant, -1 = non-bloquant.

        Returns:
            dict désérialisé, ou None si timeout.
        """
        if timeout_ms == 0:
            # Bloquant
            parts = self.sock.recv_multipart()
        else:
            if self.sock.poll(timeout_ms) == 0:
                return None
            parts = self.sock.recv_multipart()

        # parts[0] = topic, parts[1] = payload
        data = msgpack.unpackb(parts[1], object_hook=_decode_ndarray, raw=False)
        return data

    def close(self):
        self.sock.close()
        self.ctx.term()
