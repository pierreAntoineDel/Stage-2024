"""
Module pour détecter du texte et des coordonnées dans une image, traiter les reflets, et gérer les types de tâches.

Ce module comprend des fonctions pour :
- Traiter les reflets dans une image en appliquant un seuillage.
- Détecter le type de tâche en fonction des coordonnées et mettre à jour les sous-tâches et leurs positions.
- Détecter les coordonnées des mots dans une image et modifier l'image en conséquence en utilisant EasyOCR.
- Gérer la détection des lignes à l'aide de la transformation de Hough.

Dépendances :
- os
- cv2 (OpenCV)
- easyocr
- matplotlib.pyplot
- numpy
- functions.detecter_liens.detecterLignesPlanning

Fonctions :
- traitement_reflets_image: Traite les reflets dans l'image en appliquant un seuillage.

- detect_type: Détecte le type de tâche en fonction des coordonnées et met à jour les sous-tâches et leurs positions.

- detectCoordMots: Détecte les coordonnées des mots dans une image et modifie l'image en conséquence.

- detecter_texte_et_coordonnees: Détecte le texte et les coordonnées dans une image.
"""

import os
import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
from functions.detecter_liens import detecterLignesPlanning


def traitement_reflets_image(image: np.ndarray, grey: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Traite les reflets dans l'image en appliquant un seuillage.

    Args:
        image (np.ndarray): Image originale.
        grey (np.ndarray): Image en niveaux de gris.

    Returns:
        ret, grey (tuple[float, np.ndarray]): Valeur de retour du seuillage et l'image en niveaux de gris traitée.
    """

    plt.imshow(grey, cmap="gray")
    plt.show()
    ret, grey = cv2.threshold(grey, 127, 220, cv2.THRESH_BINARY)
    plt.imshow(grey, cmap="gray")
    plt.show()
    return ret, grey


def detect_type(
    ancienneTache: tuple[int, int],
    box: list[tuple[float, float]],
    center_x: float,
    center_y: float,
    posST: list[tuple[float, float]],
    sousTaches: list[str],
    mot: str,
    ancienTType: int
) -> tuple[list[str], list[tuple[float, float]], tuple[int, int], int]:
    """
    Détecte le type de tâche en fonction des coordonnées et met à jour les sous-tâches et leurs positions.

    Args:
        ancienneTache (tuple[int, int]): Position et taille de l'ancienne tâche.
        box (list[tuple[float, float]]): Coordonnées de la boîte englobante du texte détecté.
        center_x (float): Coordonnée X du centre de la boîte englobante.
        center_y (float): Coordonnée Y du centre de la boîte englobante.
        posST (list[tuple[float, float]]): Liste des positions des sous-tâches.
        sousTaches (list[str]): Liste des sous-tâches.
        mot (str): Mot détecté.
        ancienTType (int): Type de l'ancienne tâche.

    Returns:
        tuple[list[str], list[tuple[float, float]], tuple[int, int], int]:
            Mise à jour des sous-tâches, de leurs positions, de l'ancienne tâche et du type de l'ancienne tâche.
    """

    pos, taille = ancienneTache
    if (
        box[0][0] > pos + 200 and (box[3][1] - box[0][1]) < taille - 15 and box[0][1] > 300
    ):
        if not (center_x, center_y) in posST:
            sousTaches.append(mot)
            posST.append((center_x, center_y))
        ancienTType = 1
    elif (
        ancienTType == 1 and abs(box[0][0] - pos) < 10 and abs
        (
            (box[3][1] - box[0][1]) - taille
        ) < 8 and box[0][1] > 300
    ):
        if not (center_x, center_y) in posST:
            sousTaches.append(mot)
            posST.append((center_x, center_y))
    else:
        ancienTType = 0
    return sousTaches, posST, ancienneTache, ancienTType


def detectCoordMots(
    image: np.ndarray,
    recup_data: bool,
    grey: np.ndarray,
    results: list[list[Any]],
    mots_et_coordonnees: dict[str, dict[str, tuple[float, float]]],
    ancienneTache: tuple[int, int],
    total_center_x: float,
    count: int,
    modified_image: np.ndarray,
    center_color: list[int],
    vertical_line: int,
    horizontal_line: int
) -> tuple[list[Any], int, float, tuple[int, int], dict[str, dict[str, tuple[float, float]]], np.ndarray]:
    """
    Détecte les coordonnées des mots dans une image et modifie l'image en conséquence.

    Args:
        image (np.ndarray): Image d'entrée.
        recup_data (bool): Indique s'il faut récupérer les données de texte détecté.
        grey (np.ndarray): Image convertie en niveaux de gris.
        results (list[list[Any]]): Résultats de la détection de texte pour chaque lecteur.
        mots_et_coordonnees (dict[str, dict[str, tuple[float, float]]]): Dictionnaire pour stocker les mots détectés
            et leurs coordonnées.
        ancienneTache (tuple[int, int]): Coordonnées de la dernière tâche.
        total_center_x (float): Somme des coordonnées x des centres des mots.
        count (int): Nombre total de mots détectés.
        modified_image (np.ndarray): Image modifiée.
        center_color (list[int]): Couleur du centre de l'image.
        vertical_line (int): Coordonnée de la ligne verticale de référence.
        horizontal_line (int): Coordonnée de la ligne horizontale de référence.

    Returns:
        tuple[list[Any], int, float, tuple[int, int], dict[str, dict[str, tuple[float, float]]], np.ndarray]:
            Positions des sous-tâches, nombre total de mots, somme des coordonnées x des centres, coordonnées de la
            dernière tâche, dictionnaire des mots et coordonnées détectés, et image modifiée.
    """
    posST = []
    ancienTType = 0
    sousTaches = []
    for detection in results:
        box = detection[0]
        mot = detection[1]

        # Calculer les coordonnées du centre
        center_x = (box[0][0] + box[1][0] + box[2][0] + box[3][0]) / 4
        center_y = (box[0][1] + box[1][1] + box[2][1] + box[3][1]) / 4

        # Ajouter les coordonnées à la somme totale
        total_center_x += center_x
        count += 1

        # Obtenir le point le plus à droite de la boîte englobante
        rightmost_point_x = max(box[1][0], box[2][0], box[3][0])
        rightmost_point_y = center_y  # Prendre la coordonnée Y le centre
        mots_et_coordonnees[mot] = {
            "centre": (center_x, center_y),
            "point_droit": (rightmost_point_x, rightmost_point_y),
        }

        # Extraire les coordonnées de la boîte englobante
        x_min = int(min([coord[0] for coord in box]))
        y_min = int(min([coord[1] for coord in box]))
        x_max = int(max([coord[0] for coord in box]))
        y_max = int(max([coord[1] for coord in box]))

        if recup_data:
            # Extraire la région du mot de l'image d'origine
            mot_image = grey[y_min:y_max, x_min:x_max]
            # Chemin du dossier pour enregistrer les images temporaires
            temp_folder_path = "../data/images_mots"
            # Vérifier si le dossier existe, sinon le créer
            if not os.path.exists(temp_folder_path):
                os.makedirs(temp_folder_path)
            # Enregistrer l'image du mot temporairement dans le dossier spécifié
            mot_image_path = os.path.join(temp_folder_path, f"{mot}.png")
            if len(mot_image) != 0:
                # Écrivez l'image si elle n'est pas vide
                cv2.imwrite(mot_image_path, mot_image)

        # Remplacer la région de texte par la couleur du centre
        modified_image[y_min:y_max, x_min:x_max] = center_color
        if not ancienneTache == (0, 0):
            sousTaches, posST, ancienneTache, ancienTType = detect_type(
                ancienneTache, box, center_x, center_y, posST, sousTaches, mot, ancienTType
            )
        ancienneTache = (box[0][0], box[3][1] - box[0][1])
    return posST, count, total_center_x, ancienneTache, mots_et_coordonnees, modified_image, sousTaches


def detecter_texte_et_coordonnees(
    image: np.ndarray,
    reflets: bool | None,
    recup_data: bool,
    langue: str,
) -> tuple[
    dict[str, dict[str, tuple[float, float]]],
    list[list[Any]],
    Any,
    list[tuple[float, float]],
    Any,
    str
]:
    """
    Détecte le texte et les coordonnées dans une image.

    Args:
        image (np.ndarray): Image d'entrée.
        reflets (bool): Indique s'il y a des reflets à traiter.
        recup_data (bool): Indique s'il faut récupérer les données de texte détecté.
        langue (str): Langue à utiliser pour la détection du texte.

    Returns:
        tuple[dict[str, dict[str, tuple[float, float]]], list[list[Any]], Any, list[tuple[float, float]], Any, str]:
            Dictionnaire des mots et coordonnées détectés, résultats par lecteur, image en niveaux de gris, positions
            des sous-tâches, image modifiée, et direction de la moyenne des coordonnées des centres des mots.
    """

    # Convertir l'image en niveaux de gris
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if reflets:
        ret, grey = traitement_reflets_image(image, grey)

    # Initialiser les lecteurs EasyOCR pour différentes langues
    if langue == "Français" or langue == "fr":
        reader = easyocr.Reader(["fr"], gpu=False)
    else:
        reader = easyocr.Reader(["en"], gpu=False)

    # Détecter le texte avec le lecteur
    results = reader.readtext(grey, text_threshold=0.5)

    # Dictionnaire pour stocker les mots détectés et leurs coordonnées
    mots_et_coordonnees = {}

    ancienneTache = (0, 0)
    posST = []

    # Variables pour calculer la moyenne des coordonnées des centres
    total_center_x = 0
    count = 0

    # Copie de l'image pour modification
    modified_image = image.copy()

    # Couleur de remplissage (centre de l'image)
    center_color = image[image.shape[0] // 2, image.shape[1] // 2].tolist()

    # Comparaison des résultats
    vertical_line, horizontal_line = detecterLignesPlanning(image)

    posST, count, total_center_x, ancienneTache, mots_et_coordonnees, modified_image, sousTaches = detectCoordMots(
        image,
        recup_data,
        grey,
        results,
        mots_et_coordonnees,
        ancienneTache,
        total_center_x,
        count,
        modified_image,
        center_color,
        vertical_line,
        horizontal_line
    )

    # Calculer la moyenne des coordonnées des centres
    if count == 0:
        moyenne_coordonneesX = 2015
    else:
        moyenne_coordonneesX = total_center_x / count
    if moyenne_coordonneesX > 2016:
        direction = "gauche"
    else:
        direction = "droite"
    return mots_et_coordonnees, results, grey, posST, modified_image, direction, sousTaches
