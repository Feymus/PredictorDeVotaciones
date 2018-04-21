
# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Variables globales
"""
import os
import csv
import random

ESTA_CARPETA = os.path.dirname(os.path.abspath(__file__))
# Pueden ser cambiados por una dirección. (ejm "C:/archivo.csv")
ARCHIVO_CENSOS = os.path.join(ESTA_CARPETA, "properties.csv")
ARCHIVO_VOTOS = os.path.join(ESTA_CARPETA, "minutes.csv")
votos = []
censos = []
datos = {
    "SAN JOSE": {
        "rango": [1, 20],
        "votos": {},
        "propiedades": {}
    },
    "ALAJUELA": {
        "rango": [21, 35],
        "votos": {},
        "propiedades": {}
    },
    "CARTAGO": {
        "rango": [36, 43],
        "votos": {},
        "propiedades": {}
    },
    "HEREDIA": {
        "rango": [44, 53],
        "votos": {},
        "propiedades": {}
    },
    "GUANACASTE": {
        "rango": [54, 64],
        "votos": {},
        "propiedades": {}
    },
    "PUNTARENAS": {
        "rango": [65, 75],
        "votos": {},
        "propiedades": {}
    },
    "LIMON": {
        "rango": [76, 81],
        "votos": {},
        "propiedades": {}
    }
}

"""
Carga archivos csv a las variables globales "votos" y "censos"
como listas de listas
"""


def cargar_csv():

    global censos
    global votos

    with open(ARCHIVO_CENSOS, 'r') as csv_censos:
        csv_reader1 = csv.reader(csv_censos)

        for i in csv_reader1:
            censos += [i]

    with open(ARCHIVO_VOTOS, 'r') as csv_votos:
        csv_reader2 = csv.reader(csv_votos)

        for j in csv_reader2:
            votos += [j]

    cargar_datos()


'''
Guarda los datos de votaciones en un diccionario para un mejor acceso
a esta información
'''


def cargar_datos():

    global censos
    global votos

    for key in datos:
        primer_canton = datos[key]["rango"][0]
        ultimo_canton = datos[key]["rango"][1]+1

        for fila in range(primer_canton, ultimo_canton):
            canton = votos[fila][0]
            votos_en_canton = votos[fila][1:]
            propiedades_en_canton = censos[fila][1:]
            datos[key]["votos"][canton] = votos_en_canton
            datos[key]["propiedades"][canton] = propiedades_en_canton


"""
Retorna elemento aleatorio de una lista con una probabilidad dada
Entrada: lista con elementos a elegir de manera aleatorio
         lista con probabilidad de cada elemento
Salida: elemento aleatorio de la primera lista
"""


def random_pick(lista, probabilidad):

    x = random.uniform(0, 1)
    prob_acumulada = 0.0

    for item, prob_item in zip(lista, probabilidad):
        prob_acumulada += prob_item
        if x < prob_acumulada:
            break
    return item


'''
Retorna la cantidad de votos que hubo en cierta provincia
Entrada: string con el nombre de la provincia
Salida: total de votos en dicha provincia
'''


def votes_quantity_by_province(province):
    province_votes = datos[province]["votos"]
    total_votes = 0

    for key in datos[province]["votos"]:
        canton = datos[province]["votos"][key]
        total_votes += int(canton[15])

    return total_votes


'''
Retorna la cantidad de votos que hubo en todo el pais
'''


def votes_quatity_general():
    total_votes = 0

    for key in datos:
        total_votes += votes_quantity_by_province(key)

    return total_votes


'''
Retorna las probabilidades de que cierto votante pertenesca a cierto canton de
        cierta provincia
Entrada: cantidad de votos totales
         nombre de la provincia a evaluar
         lista donde se guardaran las probabilidades
         lista donde se guardaran los cantones
Salida: lista con probabilidades
        lista con cantones
'''


def probs_by_province(total_votes, province, probs, cantons):
    province_cantons = datos[province]["votos"]

    for key in datos[province]["votos"]:
        canton = datos[province]["votos"][key]
        cantons += [key]
        probs += [int(canton[15])/total_votes]


'''
Retorna las probabilidades de que cierto votante pertenesca a cierto canton
        sobre todos las provincias
Entrada: cantidad de votos totales
         lista donde se guardaran las probabilidades
         lista donde se guardaran los cantones
Salida: lista con probabilidades
        lista con cantones
'''


def general_probs(total_votes, probs, cantons):
    for key in datos:
        probs_by_province(total_votes, key, probs, cantons)


'''
Retorna un canton elegido al azar segun su probabilidad de que exista un
        un votante de dicho canton
Entrada: El nombre de la provincia a la que pertenece, o por el contrario nada
Salida: Un canton elegido segun provincia o segun todas las provincias
'''


def pick_canton(province="NONE"):
    if province == "NONE":

        total_votes = votes_quatity_general()

        probs = []
        cantons = []

        general_probs(total_votes, probs, cantons)

        return random_pick(cantons, probs)

    else:
        total_votes = votes_quantity_by_province(province)

        probs = []
        cantons = []

        probs_by_province(total_votes, province, probs, cantons)

        return random_pick(cantons, probs)


'''
Retorna la provincia a la que pertenece un canton
Entrada: nombre del canton con el que se busca
Salida: el nombre de la provincia a la que pertenece
'''


def search_province_by_canton(canton):
    for province_name in datos:
        for canton_name in datos[province_name]["votos"]:
            if canton_name == canton:
                return province_name
    return -1


'''
Retorna un rango de edad para una muestra
'''


def pick_age():
    age_delimiters = ["18-24", "25-64", "65 O MAS"]
    probabilities = [611228/2859287, 1935410/2859287, 312649/2859287]

    return random_pick(age_delimiters, probabilities)


'''
Retorna si la persona es depediente o no
Entrada: rango de edad (string, ver pick_age)
Salida: un string definiendo si la persona es dependiente o no
'''


def is_dependent(age_range):
    if age_range == "65 O MAS":
        return "DEPENDIENTE"
    else:
        return "PRODUCTIVO"


'''
Retorna si la persona es alfabeta o no
Entrada: rango de edad (string, ver pick_age)
Salida: un string definiendo si la persona es alfabeta o no
'''


def is_literate_pct(age_range, province, canton):
    if age_range == "18-24":
        return float(datos[province]["propiedades"][canton][11])/100
    else:
        return float(datos[province]["propiedades"][canton][12])/100


'''
Retorna el porcentaje de asistencia a educacion regular de la persona promedio
        de dicha propiedades
Entrada: rango de edad (string, ver pick_age)
Salida: un string definiendo si la persona asistio a educacion regular
'''


def regular_edu_pct(age_range, province, canton):
    if age_range == "18-24":
        return float(datos[province]["propiedades"][canton][19])/100
    else:
        return float(datos[province]["propiedades"][canton][20])/100


'''
Retorna la probabilidad de que una persona segun su genero pertenezca o no al
        grupo laboral
Entrada: genero de la persona
         nombre de la provincia
         nombre del canton
Salida: probabilidad de pertenecer
'''


def get_work_pct(gender, province, canton):

    if gender == "HOMBRE":
        part_pct_male = float(datos[province]["propiedades"][canton][23])/100
        return part_pct_male
    else:
        part_pct_female = float(datos[province]["propiedades"][canton][24])/100
        return part_pct_female


'''
Retorna el porcentaje de una persona que esta asegurada o no
Entrada: si la persona trabaja o no
         el nombre de la provincia
         el nombre del canton
Salida: probabilidad de estar asegurado
'''


def get_insured_pct(work, province, canton):
    if work == "SI":
        return 1 - (float(datos[province]["propiedades"][canton][25]) / 100)
    else:
        return 1 - (float(datos[province]["propiedades"][canton][28]) / 100)


'''
'''


def generated_vote(province, canton):

    total_votes = int(datos[province]["votos"][canton][15])
    probs = []

    for vote in datos[province]["votos"][canton]:
        prob = int(vote)/total_votes
        probs += [prob]

    probs = probs[:15]
    options = [
        "ACCESIBILIDAD SIN EXCLUSION", "ACCION CIUDADANA",
        "ALIANZA DEMOCRATA CRISTIANA", "DE LOS TRABAJADORES", "FRENTE AMPLIO",
        "INTEGRACION NACIONAL", "LIBERACION NACIONAL", "MOVIMIENTO LIBERTARIO",
        "NUEVA GENERACION", "RENOVACION COSTARRICENSE",
        "REPUBLICANO SOCIAL CRISTIANO", "RESTAURACION NACIONAL",
        "UNIDAD SOCIAL CRISTIANA", "NULO", "BLANCO"
    ]

    return random_pick(options, probs)


'''
Retorna una muestra de una provincia y canton particulares
Entrada: nombre de la provincias
         nombre del canton
Salida: una lista con los datos de la muestra
'''


def generate_sample_by_province(province, canton):
    sample = []

    total_population = datos[province]["propiedades"][canton][0]
    surface = datos[province]["propiedades"][canton][1]
    density = datos[province]["propiedades"][canton][2]

    urban_pct = float(datos[province]["propiedades"][canton][3]) / 100
    rural_pct = 1-urban_pct
    urban = random_pick(
        ["URBANO", "RURAL"],
        [urban_pct, rural_pct]
    )

    males_relation = float(datos[province]["propiedades"][canton][4])
    male_pct = males_relation / (males_relation + 100)
    female_pct = 1-male_pct
    gender = random_pick(
        ["HOMBRE", "MUJER"],
        [male_pct, female_pct]
    )

    age = pick_age()
    dependent = is_dependent(age)

    literate_pct = is_literate_pct(age, province, canton)
    not_literate_pct = 1-literate_pct
    literate = random_pick(
        ["ALFABETA", "NO ALFABETA"],
        [literate_pct, not_literate_pct]
    )

    avg_scholarship = datos[province]["propiedades"][canton][13]

    regular_pct = regular_edu_pct(age, province, canton)
    not_regular_edu_pct = 1-regular_pct
    regular_edu = random_pick(
        ["SI", "NO"],
        [regular_pct, not_regular_edu_pct]
    )

    work_pct = get_work_pct(gender, province, canton)
    not_work_pct = 1-work_pct
    work = random_pick(
        ["SI", "NO"],
        [work_pct, not_work_pct]
    )

    insured_pct = get_insured_pct(work, province, canton)
    not_insured_pct = 1-insured_pct
    insured = random_pick(
        ["SI", "NO"],
        [insured_pct, not_insured_pct]
    )

    individual_houses = datos[province]["propiedades"][canton][6]
    occupants_avg = datos[province]["propiedades"][canton][7]

    good_condition_pct = float(datos[province]["propiedades"][canton][8]) / 100
    bad_condition_pct = 1-good_condition_pct
    condition = random_pick(
        ["BUEN ESTADO", "MAL ESTADO"],
        [good_condition_pct, bad_condition_pct]
    )

    crowded_pct = float(datos[province]["propiedades"][canton][9]) / 100
    not_crowded_pct = 1-crowded_pct
    crowded = random_pick(
        ["HACINADA", "NO HACINADA"],
        [crowded_pct, not_crowded_pct]
    )

    born_abroad_pct = float(datos[province]["propiedades"][canton][26]) / 100
    not_born_abroad_pct = 1-born_abroad_pct
    born_abroad = random_pick(
        ["EXTRANJERO", "NACIONAL"],
        [born_abroad_pct, not_born_abroad_pct]
    )

    handicapped_pct = float(datos[province]["propiedades"][canton][27]) / 100
    not_handicapped_pct = 1-handicapped_pct
    handicapped = random_pick(
        ["DISCAPACITADO", "NO DISCAPACITADO"],
        [handicapped_pct, not_handicapped_pct]
    )

    female_head_pct = float(datos[province]["propiedades"][canton][29]) / 100
    not_female_head_pct = 1-female_head_pct
    female_head = random_pick(
        ["SI", "NO"],
        [female_head_pct, not_female_head_pct]
    )

    shared_head_pct = float(datos[province]["propiedades"][canton][30]) / 100
    not_shared_head_pct = 1-shared_head_pct
    shared_head = random_pick(
        ["SI", "NO"],
        [shared_head_pct, not_shared_head_pct]
    )

    vote = generated_vote(province, canton)

    sample += [
        province, canton, total_population, surface, density, urban,
        gender, age, dependent, literate, avg_scholarship, regular_edu, work,
        insured, individual_houses, occupants_avg, condition, crowded,
        born_abroad, handicapped, female_head, shared_head, vote
    ]

    return sample


'''
Retorna una muestra ya sea para una provincia y canton particulares o de forma
aleatoria
Entrada: nombre de la provincia de la cual generar la muestra (opcional)
Salida: lista con la muestra generada
'''


def generate_sample(province="NONE"):
    if province == "NONE":
        canton = pick_canton()
        province = search_province_by_canton(canton)
        return generate_sample_by_province(province, canton)
    else:
        canton = pick_canton(province)
        return generate_sample_by_province(province, canton)


'''
Retorna una lista con diferentes muestras de votantes generadas aleatoriamente
Entrada: cantidad de muestras a generar
Salida: lista con las muestras
'''


def generar_muestra_pais(n):
    cargar_csv()
    muestras = []

    for muestra in range(0, n):
        muestras += [generate_sample()]

    return muestras


'''
Retorna una lista con diferentes muestras de votantes generadas aleatoriamente
        para cierta provincia
Entrada: cantidad de muestras a generar
         nombre de la provincia
Salida: lista con las muestras
'''


def generar_muestra_provincia(n, nombre_provincia):
    nombre_provincia = nombre_provincia.upper()
    cargar_csv()
    muestras = []

    for muestra in range(0, n):
        muestras += [generate_sample(nombre_provincia)]

    return muestras


'''
Retorna un csv con todas las muestras que se le pasen por parametro
Entrada: una lista de listas con cada muestra
Salida: un csv con todas las muestras
'''


def pasar_a_csv(muestras):

    with open("./muestras.csv", "w", newline='') as file:

        writer = csv.writer(file, delimiter=",")

        for muestra in muestras:
            writer.writerow(muestra)


def porcentajes(muestras):
    total_urbano = 0
    total_rural = 0
    total_hombres = 0
    total_mujeres = 0
    total_dependientes = 0
    total_independientes = 0
    total_buen_estado = 0
    total_mal_estado = 0
    total_hacinadas = 0
    total_no_hacinadas = 0
    total_alfabetas = 0
    total_no_alfabetas = 0
    total_ed_regular = 0
    total_no_ed_regular = 0
    total_ed_18_24 = 0
    total_no_ed_18_24 = 0
    total_ed_25_mas = 0
    total_no_ed_25_mas = 0
    total_trabajando = 0
    total_no_trabajando = 0
    total_mujeres_tjo = 0
    total_mujeres_no = 0
    total_hombres_tjo = 0
    total_hombres_no = 0
    total_asegurado = 0
    total_no_asegurado = 0
    total_no_ago_con_tjo = 0
    total_extranjero = 0
    total_nacional = 0
    total_discapacitado = 0
    total_no_discap = 0
    total_jef_femenina = 0
    total_jef_compartida = 0

    for muestra in muestras:
        if muestra[5] == "URBANO":
            total_urbano += 1
        else:
            total_rural += 1

        if muestra[6] == "MUJER":
            total_mujeres += 1
        else:
            total_hombres += 1

        if muestra[8] == "PRODUCTIVO":
            total_independientes += 1
        else:
            total_dependientes += 1

        if muestra[16] == "BUEN ESTADO":
            total_buen_estado += 1
        else:
            total_mal_estado += 1

        if muestra[17] == "HACINADA":
            total_hacinadas += 1
        else:
            total_no_hacinadas += 1

        if muestra[9] == "ALFABETA":
            total_alfabetas += 1
        else:
            total_no_alfabetas += 1

        if muestra[11] == "SI":
            total_ed_regular += 1
            if muestra[7] == "18-24":
                total_ed_18_24 += 1
            else:
                total_ed_25_mas += 1
        else:
            total_no_ed_regular += 1
            if muestra[7] == "18-24":
                total_no_ed_18_24 += 1
            else:
                total_no_ed_25_mas += 1

        if muestra[12] == "SI":
            total_trabajando += 1
            if muestra[6] == "MUJER":
                total_mujeres_tjo += 1
            else:
                total_hombres_tjo += 1
        else:
            total_no_trabajando += 1
            if muestra[6] == "MUJER":
                total_mujeres_no += 1
            else:
                total_hombres_no += 1

        if muestra[13] == "SI":
            total_asegurado += 1
        else:
            total_no_asegurado += 1
            if muestra[12] == "SI":
                total_no_ago_con_tjo += 1

        if muestra[18] == "EXTRANJERO":
            total_extranjero += 1
        else:
            total_nacional += 1

        if muestra[19] == "DISCAPACITADO":
            total_discapacitado += 1
        else:
            total_no_discap += 1

        if muestra[20] == "SI":
            total_jef_femenina += 1

        if muestra[21] == "SI":
            total_jef_compartida += 1

    pct_urbano = total_urbano/(total_urbano+total_rural)
    pct_genero_fem = total_mujeres/(total_mujeres+total_hombres)
    pct_depend = total_dependientes/(total_dependientes+total_independientes)
    pct_estado = total_buen_estado/(total_buen_estado+total_mal_estado)
    pct_hac = total_hacinadas/(total_hacinadas+total_no_hacinadas)
    pct_alfabeta = total_alfabetas/(total_alfabetas+total_no_alfabetas)
    pct_ed_regular = total_ed_regular/(total_ed_regular+total_no_ed_regular)
    pct_trabajando = total_trabajando/(total_trabajando+total_no_trabajando)
    pct_mujeres_tjo = total_mujeres_tjo/(total_mujeres_tjo+total_mujeres_no)
    pct_hombres_tjo = total_hombres_tjo/(total_hombres_tjo+total_hombres_no)
    pct_no_asegurado = total_no_asegurado/(total_no_asegurado+total_asegurado)
    pct_no_ago_con_tjo = total_no_ago_con_tjo/total_trabajando
    pct_extranjero = total_extranjero/(total_extranjero+total_nacional)
    pct_discap = total_discapacitado/(total_discapacitado+total_no_discap)
    pct_jef_fem = total_jef_femenina/len(muestras)
    pct_jef_comp = total_jef_compartida/len(muestras)
    pct_ed_18_24 = total_ed_18_24/(total_ed_18_24+total_no_ed_18_24)
    pct_ed_25_mas = total_ed_25_mas/(total_ed_25_mas+total_no_ed_25_mas)

    return (
        pct_urbano, pct_genero_fem, pct_depend, pct_estado, pct_hac,
        pct_alfabeta, pct_ed_regular, pct_trabajando, pct_mujeres_tjo,
        pct_hombres_tjo, pct_no_asegurado, pct_no_ago_con_tjo, pct_extranjero,
        pct_discap, pct_jef_fem, pct_jef_comp, pct_ed_18_24, pct_ed_25_mas
    )


def sacar_promedios(muestras):
    promedios = {
        "POBLACION URBANA": 0,
        "MUJERES": 0,
        "HOMBRES": 0,
        "DEPENDIENTES": 0,
        "VIVIENDAS BE": 0,
        "VIVIENDAS H": 0,
        "ALFABETISMO": 0,
        "EDUCACION RE": 0,
        "EDUC 18-24": 0,
        "EDUC 25 MAS": 0,
        "TRABAJO": 0,
        "TRABAJO M": 0,
        "TRABAJO H": 0,
        "NO ASEGURADO": 0,
        "NO ASEGURADO CON TRABAJO": 0,
        "EXTRANJERO": 0,
        "DISCAPACIDAD": 0,
        "JEFATURA F": 0,
        "JEFATURA C": 0
    }

    pcts = porcentajes(muestras)
    promedios["POBLACION URBANA"] = pcts[0]
    promedios["MUJERES"] = pcts[1]
    promedios["HOMBRES"] = 1-pcts[1]
    promedios["DEPENDIENTES"] = pcts[2]
    promedios["VIVIENDAS BE"] = pcts[3]
    promedios["VIVIENDAS H"] = pcts[4]
    promedios["ALFABETISMO"] = pcts[5]
    promedios["EDUCACION RE"] = pcts[6]
    promedios["TRABAJO"] = pcts[7]
    promedios["TRABAJO M"] = pcts[8]
    promedios["TRABAJO H"] = pcts[9]
    promedios["NO ASEGURADO"] = pcts[10]
    promedios["NO ASEGURADO CON TRABAJO"] = pcts[11]
    promedios["EXTRANJERO"] = pcts[12]
    promedios["DISCAPACIDAD"] = pcts[13]
    promedios["JEFATURA F"] = pcts[14]
    promedios["JEFATURA C"] = pcts[15]
    promedios["EDUC 18-24"] = pcts[16]
    promedios["EDUC 25 MAS"] = pcts[17]

    return promedios


def main():
    indicadores = [
        "Provincia", "Canton", "Total de la población", "Superficie",
        "Densidad de la población", "Urbano/Rural", "Género", "Edad",
        "Dependencia", "Alfabeta", "Escolaridad promedio",
        "Escolaridad regular", "Trabaja", "Asegurado",
        "Cant. casas individuales", "Ocupantes promedio", "Condicion",
        "Hacinada", "Nacido en...", "Discapacitado", "Jefatura femenina",
        "Jefatura compartida", "Voto"
    ]

    muestras = generar_muestra_pais(25000)


if __name__ == '__main__':
    main()
