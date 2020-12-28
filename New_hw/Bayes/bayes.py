import pandas as pd
import numpy as np


# Наивный Байесовкий классификатор


def bayes_classifier(disease, symptom, our_symptoms):
    disease_probes = []
    all_p = disease['количество пациентов'].values[-1]

    for i in disease['количество пациентов'].values:
        if i == all_p:
            continue
        disease_probes.append(i / all_p)

    our_probes = [1] * (len(disease['Болезнь']) - 1)

    for i in range(len(disease['Болезнь']) - 1):
        our_probes[i] *= disease_probes[i]
        for j in range(len(symptom) - 1):
            if our_symptoms[j] == 1:
                our_probes[i] *= float(str(symptom.iloc[j][i + 1]).replace(',', '.'))

    max_index = 0
    max_value = our_probes[max_index]
    for i in range(1, len(our_probes)):
        if max_value < our_probes[i]:
            max_value = our_probes[i]
            max_index = i
    return disease['Болезнь'][max_index]


if __name__ == '__main__':
    # Симптомы
    dataset_symptom = pd.read_csv('../datasets/symptom.csv', ';')

    # Болезни
    dataset_disease = pd.read_csv('../datasets/disease.csv', ';')

    print(dataset_symptom)
    print(dataset_disease)

    # Преобразовываем набор наших данных в таблицу частот (map)
    P = dict(zip(dataset_disease['Болезнь'], dataset_disease['количество пациентов']
                 / dataset_disease['количество пациентов'][len(dataset_disease['количество пациентов']) - 1]))

    print(P)

    # Генерируем рандомный список симптомов
    list_symptoms = [np.random.randint(0, 2) for i in range(len(dataset_symptom) - 1)]
    print("\nСписок симптомов: ")
    print(list_symptoms)

    result = bayes_classifier(dataset_disease, dataset_symptom, list_symptoms)
    print("\nРезультат: " + result)
