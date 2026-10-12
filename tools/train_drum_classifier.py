#!/usr/bin/env python3
"""Trains a KNN classifier on training_data.csv (see collect_training_data.py)
to distinguish kick/snare/hihat by timbre - lightweight classical ML,
following the SMC-LAB streaming drum transcription approach (onset -> short
spectral feature vector -> KNN classifier), no neural network needed."""

import csv
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "training_data.csv"
MODEL_PATH = ROOT / "models" / "drum_knn.pkl"


def main():
    if not DATA_PATH.exists():
        print(f"Keine Trainingsdaten gefunden ({DATA_PATH}). Erst collect_training_data.py fuer kick/snare/hihat laufen lassen.")
        sys.exit(1)

    rows = list(csv.reader(open(DATA_PATH)))
    data = rows[1:]
    X = np.array([[float(v) for v in row[:-1]] for row in data])
    y = np.array([row[-1] for row in data])

    counts = {label: int(np.sum(y == label)) for label in sorted(set(y))}
    print(f"Trainingsdaten: {len(y)} Treffer total - {counts}")
    if len(y) < 30 or len(set(y)) < 2:
        print("Zu wenige oder zu einseitige Daten fuer sinnvolles Training - mehr Treffer pro Klasse aufnehmen.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    scaler = StandardScaler().fit(X_train)
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(scaler.transform(X_train), y_train)

    y_pred = clf.predict(scaler.transform(X_test))
    print("\n" + classification_report(y_test, y_pred, zero_division=0))
    labels_sorted = sorted(set(y))
    print(f"Confusion matrix (Zeilen=echt, Spalten=vorhergesagt) {labels_sorted}:")
    print(confusion_matrix(y_test, y_pred, labels=labels_sorted))

    # Re-fit on ALL data for the deployed model - the split above was only
    # for an honest accuracy estimate.
    scaler_full = StandardScaler().fit(X)
    clf_full = KNeighborsClassifier(n_neighbors=5)
    clf_full.fit(scaler_full.transform(X), y)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"scaler": scaler_full, "clf": clf_full}, MODEL_PATH)
    print(f"\nModell gespeichert: {MODEL_PATH}")


if __name__ == "__main__":
    main()
