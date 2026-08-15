# -*- coding: utf-8 -*- TEste convergencia
"""
================================================================================
Grupo 3 - MET-579 -- TESTE DE CONVERGENCIA DE GRADE (Lambda_z fisico fixo)
================================================================================
Pergunta a responder: "Cada esquema converge para a solucao analitica quando
Delta z diminui? O refinamento da grade melhora a solucao para qualquer
inclinacao da onda (k/m)?"

--------------------------------------------------------------------------------
CONFIGURACAO (conforme especificado)
--------------------------------------------------------------------------------
Ao contrario dos experimentos anteriores (onde Delta z era fixo e variavamos
PPWz mudando a escala da onda), aqui e' o oposto: fixamos o COMPRIMENTO DE
ONDA VERTICAL FISICO,

    Lambda_z = 2000 m   =>   m = 2*pi/Lambda_z   (fixo, nao muda com a grade)

e variamos gradualmente Delta z (isto e', refinamos a grade). Como
PPWz = Lambda_z/Delta z, o numero de pontos por comprimento de onda muda
automaticamente conforme a tabela:

    Delta z [m]   PPWz
       500          4
       400          5
       333.33       6
       250          8
       200         10
       166.67      12
       125         16

Para cada Delta z, calculamos theta = m*Delta z e a frequencia numerica
omega_num (Eq. 3.3) para os 4 esquemas, comparando com a frequencia analitica
omega_analitica (Eq. 2.5) -- ambas para o MESMO par fisico (k,m), com m fixo
e k = (k/m)*m.

Fazemos isso para tres razoes k/m (tres "inclinacoes" de onda diferentes):
    k/m = 1.0   (onda a 45 graus da vertical)
    k/m = 0.5   (onda mais "vertical"/inclinada, m domina)
    k/m = 2.0   (onda mais "deitada"/horizontal, k domina)

f=0 e U=0 (mesmas simplificacoes do experimento anterior, isolando o efeito
puro de resolucao/esquema).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0) Parametros fisicos fixos
# ------------------------------------------------------------------
N_BV = 0.012
Lambda_z = 2000.0
m_fixo = 2.0 * np.pi / Lambda_z   # numero de onda vertical FISICO, fixo

# ------------------------------------------------------------------
# 1) Delta z's da tabela (PPWz muda automaticamente = Lambda_z/Delta z)
# ------------------------------------------------------------------
dz_lista = np.array([500.0, 400.0, 1000.0 / 3.0, 250.0, 200.0, 500.0 / 3.0, 125.0])
ppwz_lista = Lambda_z / dz_lista   # 4, 5, 6, 8, 10, 12, 16

# ------------------------------------------------------------------
# 2) Numeros de onda modificados (m*Delta z), identicos aos experimentos
#    anteriores
# ------------------------------------------------------------------
def mstar_dz_2a_ordem(theta):
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    return 2.0 * np.sin(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": dict(func=mstar_dz_2a_ordem, cor="#c0392b", estilo="--"),
    "Nao-alternada, 4a ordem": dict(func=mstar_dz_4a_ordem, cor="#2980b9", estilo="-."),
    "Compacto (Pade), 4a ordem": dict(func=mstar_dz_compacto, cor="#27ae60", estilo=":"),
    "Alternada (Lorenz), 2a ordem": dict(func=mstar_dz_staggered_2a_ordem, cor="#8e44ad", estilo="-"),
}


# ------------------------------------------------------------------
# 3) Relacoes de dispersao (f=0, U=0 -- caso puro da Parte 1)
# ------------------------------------------------------------------
def omega_analitica(k, m, N=N_BV):
    return N * np.abs(k) / np.sqrt(k**2 + m**2)


def omega_numerica(k, m, dz, func_mstar, N=N_BV):
    theta = m * dz
    mstar = func_mstar(theta) / dz
    return N * np.abs(k) / np.sqrt(k**2 + mstar**2)


# ------------------------------------------------------------------
# 4) Varredura: para cada razao k/m, calcula o erro relativo em omega
#    para cada Delta z da tabela e cada esquema
# ------------------------------------------------------------------
razoes_km = [1.0, 0.5, 2.0]

resultados = {}  # resultados[razao][nome_esquema] = array de erros (um por dz)

for razao in razoes_km:
    k_fixo = razao * m_fixo
    resultados[razao] = {}
    for nome, cfg in ESQUEMAS.items():
        erros = []
        for dz in dz_lista:
            omega_n = omega_numerica(k_fixo, m_fixo, dz, cfg["func"])
            omega_a = omega_analitica(k_fixo, m_fixo)
            erros.append(abs(omega_n - omega_a) / omega_a)
        resultados[razao][nome] = np.array(erros)

# ------------------------------------------------------------------
# 5) Tabelas no console
# ------------------------------------------------------------------
print(f"Lambda_z fixo = {Lambda_z:.0f} m  =>  m = {m_fixo:.4e} rad/m\n")
for razao in razoes_km:
    print(f"--- k/m = {razao}  (k = {razao*m_fixo:.4e} rad/m) ---")
    cabecalho = f"{'Delta z [m]':>12s}{'PPWz':>7s}"
    for nome in ESQUEMAS:
        cabecalho += f"{nome:>28s}"
    print(cabecalho)
    for i, dz in enumerate(dz_lista):
        linha = f"{dz:12.2f}{ppwz_lista[i]:7.1f}"
        for nome in ESQUEMAS:
            linha += f"{resultados[razao][nome][i]*100:27.3f}%"
        print(linha)
    print()

# ------------------------------------------------------------------
# 6) Figura: 3 paineis lado a lado (um por razao k/m), erro vs. PPWz
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), sharey=True)

for ax, razao in zip(axes, razoes_km):
    for nome, cfg in ESQUEMAS.items():
        ax.plot(ppwz_lista, resultados[razao][nome] * 100, cfg["estilo"],
                 color=cfg["cor"], marker="o", ms=5, lw=1.8, label=nome)
    ax.set_yscale("log")
    ax.set_xlabel("PPWz  (aumenta = grade mais refinada)")
    ax.set_title(f"k/m = {razao}", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.3, which="both")

axes[0].set_ylabel("Erro relativo em $\\omega$ [%]  (escala log)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.08))

fig.suptitle(
    r"Convergência de grade com $\Lambda_z$=2000 m fixo (m fixo), $\Delta z$ variando"
    "\npara três inclinações de onda -- Grupo 3, MET-579",
    fontsize=13, fontweight="bold", y=1.03,
)
fig.tight_layout()
fig.savefig("convergencia_lambda_fixo.png", dpi=150, bbox_inches="tight")
print("Figura salva em convergencia_lambda_fixo.png")

# ------------------------------------------------------------------
# 7) Resposta objetiva as duas perguntas do enunciado
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESPOSTA A PERGUNTA: cada esquema converge quando Delta z diminui?")
for nome in ESQUEMAS:
    e_grosseiro = resultados[1.0][nome][0]   # Delta z=500 (PPWz=4)
    e_fino = resultados[1.0][nome][-1]        # Delta z=125 (PPWz=16)
    print(f"  {nome:32s}: erro cai de {e_grosseiro*100:6.2f}% (PPWz=4) "
          f"para {e_fino*100:6.3f}% (PPWz=16)  ->  {'CONVERGE' if e_fino < e_grosseiro else 'NAO CONVERGE'}")

print("\nRESPOSTA A PERGUNTA: o refinamento ajuda para qualquer inclinacao (k/m)?")
for razao in razoes_km:
    maior_erro_fino = max(resultados[razao][nome][-1] for nome in ESQUEMAS)
    maior_erro_grosseiro = max(resultados[razao][nome][0] for nome in ESQUEMAS)
    print(f"  k/m={razao}: pior erro cai de {maior_erro_grosseiro*100:.2f}% (PPWz=4) "
          f"para {maior_erro_fino*100:.3f}% (PPWz=16)")
print("=" * 70)