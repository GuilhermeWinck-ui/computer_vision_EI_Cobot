"""
zmq_test_sub.py — Testeur Subscriber ZeroMQ
=============================================
À COMPLÉTER — Étape 0, TP1

Objectif : s'abonner au publisher zmq_test_pub.py, recevoir les messages
           et afficher les informations clés.

Lancez d'abord zmq_test_pub.py dans un terminal, puis ce script dans un autre.

Usage :
    python zmq_test_sub.py
    python zmq_test_sub.py --port 5556 --topic test/hello
"""

import argparse
import time
from zmq_utils import Subscriber


def main(args):
    # TODO 1 : Créer un Subscriber connecté à args.host:args.port sur args.topic
    #           Regardez la docstring de la classe Subscriber dans zmq_utils.py
    sub = None  # <-- remplacer par l'instanciation correcte

    print("En attente de messages... (Ctrl+C pour arrêter)")

    try:
        while True:
            # TODO 2 : Recevoir un message avec un timeout de 1000 ms
            #           Si aucun message reçu (data is None), afficher "..." et continuer
            data = None  # <-- remplacer par sub.receive(...)

            if data is None:
                print("...")
                continue

            # TODO 3 : Afficher les informations suivantes :
            #   - Le numéro du message (clé "counter")
            #   - Le timestamp formaté en heure lisible (indice : time.strftime)
            #   - Le shape du tableau numpy "frame" (attribut .shape)
            #   - Le nombre de markers détectés (len de la liste "markers")
            #   - L'id et le center de chaque marker
            #
            # Exemple de sortie attendue :
            #   [SUB] Message #3 | 14:32:05 | frame=(480, 848, 3) | 2 marker(s)
            #       → Marker id=0  center=[320, 240]
            #       → Marker id=1  center=[100, 150]

            pass  # <-- remplacer par votre affichage

    except KeyboardInterrupt:
        print("\n[SUB] Arrêt.")
    finally:
        # TODO 4 : Fermer proprement le subscriber (méthode close())
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  type=str, default="localhost")
    parser.add_argument("--port",  type=int, default=5555)
    parser.add_argument("--topic", type=str, default="vision")
    main(parser.parse_args())
