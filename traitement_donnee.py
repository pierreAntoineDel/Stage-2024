"""
Ce module contient des fonctions pour préparer des données textuelles pour un modèle BERT.
Les fonctions incluent la suppression des valeurs manquantes, le calcul de la longueur maximale des tokens,
la tokenisation des données, la conversion en tenseurs PyTorch, et la division des données en ensembles d'entraînement
et de test.

Modules requis:
    - transformers: pour le tokenizer BERT
    - torch: pour les tenseurs PyTorch
    - sklearn: pour la division des données en ensembles d'entraînement et de test
    - pandas: pour la manipulation des données
"""

from transformers import BertTokenizer
import torch
from sklearn.model_selection import train_test_split
import pandas as pd


def calculate_max_length(
    input_data: list[list[str]], output_data: list[list[str]]
) -> tuple[list[list[str]], list[list[str]], BertTokenizer]:
    """
    Calculer la longueur maximale des tokens dans les données d'entrée et de sortie.

    Args:
        input_data (List[List[str]]): Données d'entrée.
        output_data (List[List[str]]): Données de sortie.

    Returns:
        Tuple contenant les données d'entrée nettoyées, les données de sortie nettoyées, et le tokenizer BERT.
    """

    # Suppression des 'nan' (manques de donnees)
    input_data_cleaned = [['' if pd.isna(value) else value for value in row] for row in input_data]
    output_data_cleaned = [['' if pd.isna(value) else value for value in row] for row in output_data]

    # Charger le tokenizer BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    max_token_length = 0

    # Parcourir toutes les données input
    for text in input_data_cleaned:
        # Tokenisation
        text = str(text)
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_token_length:
            max_token_length = len(tokens)

    # Utiliser la longueur maximale calculée comme valeur de max_length
    max_length_input = max_token_length
    print("Max_length input:", max_length_input)

    # Parcourir toutes les données output
    for text in output_data_cleaned:
        # Tokenisation
        text = str(text)
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_token_length:
            max_token_length = len(tokens)

    # Utiliser la longueur maximale calculée comme valeur de max_length
    max_length_output = max_token_length
    print("Max_length output:", max_length_output)

    return input_data_cleaned, output_data_cleaned, tokenizer


def tokenize_data(input_data: list[str], output_data: list[str], tokenizer: BertTokenizer) -> tuple[list[dict], list[dict]]:
    """
    Tokeniser les données d'entrée et de sortie.

    Args:
        input_data (List[str]): Données d'entrée.
        output_data (List[str]): Données de sortie.
        tokenizer (BertTokenizer): Tokenizer BERT.

    Returns:
        Tuple contenant les listes de tokens pour les données d'entrée et de sortie.
    """

    input_tokens_list = []
    output_tokens_list = []

    # Parcourir toutes les données input
    for text in input_data:
        # Tokenisation
        text = str(text)

        tokensIN = tokenizer.encode_plus(
            text,                      # Texte à encoder
            add_special_tokens=True,   # Ajouter [CLS] et [SEP]
            max_length=200,            # Longueur maximale de la séquence de sortie
            padding="max_length",      # Remplir ou tronquer au maximum
            truncation=True,           # Tronquer les séquences si elles dépassent max_length
            return_tensors="pt"        # Retourner des tensors PyTorch
        )

        input_tokens_list.append(tokensIN)

    # Parcourir toutes les données output
    for text in output_data:
        # Tokenisation
        text = str(text)

        tokensOUT = tokenizer.encode_plus(
            text,                      # Texte à encoder
            add_special_tokens=True,   # Ajouter [CLS] et [SEP]
            max_length=512,            # Longueur maximale de la séquence de sortie
            padding="max_length",      # Remplir ou tronquer au maximum
            truncation=True,           # Tronquer les séquences si elles dépassent max_length
            return_tensors="pt"        # Retourner des tensors PyTorch
        )

        output_tokens_list.append(tokensOUT)

    return input_tokens_list, output_tokens_list


def convert_to_tensors(input_tokens_list: list[dict], output_tokens_list: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convertir les listes de tokens en tenseurs PyTorch.

    Args:
        input_tokens_list (List[dict]): Liste des tokens d'entrée.
        output_tokens_list (List[dict]): Liste des tokens de sortie.

    Returns:
        Tuple contenant les tenseurs PyTorch des données d'entrée et de sortie.
    """

    input_tensor_list = []
    output_tensor_list = []

    # Parcourir les données input
    for tokens in input_tokens_list:
        input_tensor_list.append(tokens['input_ids'])

    # Parcourir les données output
    for tokens in output_tokens_list:
        output_tensor_list.append(tokens['input_ids'])

    # Convertir les listes de tenseurs en tensors PyTorch
    input_tensors = torch.stack(input_tensor_list, dim=0)
    output_tensors = torch.stack(output_tensor_list, dim=0)

    return input_tensors, output_tensors


def split_train_test(
    input_tensors: torch.Tensor, output_tensors: torch.Tensor, test_size: float = 0.2, random_state: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Diviser les tenseurs en ensembles d'entraînement et de test.

    Args:
        input_tensors (torch.Tensor): Tenseurs des données d'entrée.
        output_tensors (torch.Tensor): Tenseurs des données de sortie.
        test_size (float): Proportion des données à inclure dans l'ensemble de test.
        random_state (int): État aléatoire pour la reproductibilité.

    Returns:
        Tuple contenant les tenseurs d'entraînement et de test pour les données d'entrée et de sortie.
    """

    # Diviser les données en ensembles d'entraînement et de test
    input_train, input_test, output_train, output_test = train_test_split(
        input_tensors, output_tensors,
        test_size=test_size,
        random_state=random_state
    )
    return input_train, input_test, output_train, output_test
