# -*- coding: utf-8 -*- contrastes entre esquemas
"""
================================================================================
Grupo 3 - MET-579 -- EXPERIMENTO 3: CONTRASTES ENTRE ESQUEMAS
================================================================================
Em vez de comparar os 4 esquemas todos de uma vez contra a analitica (o que
so' permite concluir "esquema X ficou mais perto"), este experimento isola
TRES efeitos distintos, cada um respondendo a uma pergunta especifica sobre
POR QUE um esquema e' melhor que outro:

  Contraste A - EFEITO DA ORDEM FORMAL
      Nao-alternada 2a ordem  vs  Nao-alternada 4a ordem
      (mesma posicao de variaveis -- grade nao-alternada -- muda so' a ordem
      do stencil). Responde: quanto se ganha SO' aumentando a ordem?

  Contraste B - EFEITO DA POSICAO DAS VARIAVEIS (staggering)
      Nao-alternada 2a ordem  vs  Alternada (Lorenz) 2a ordem
      (mesma ordem formal -- 2a -- muda so' a posicao/alternancia da grade).
      Responde: quanto se ganha SO' alternando a grade, sem aumentar a ordem?

  Contraste C - EFEITO DA COMPACTACAO
      Nao-alternada 4a ordem (explicita)  vs  Compacta (Pade) 4a ordem
      (mesma ordem formal -- 4a -- um e' explicito, outro e' implicito/
      tridiagonal). Responde: esquemas de mesma ordem formal tem a mesma
      qualidade espectral?

Metrica: erro relativo em m* (|m*/m - 1|), a mesma usada na Tabela 3.1 da
Parte 1 -- e' a metrica "pura" de fidelidade do operador discreto, que nao
depende da razao k/m escolhida (ao contrario do erro em omega).

Avaliado em PPWz = 4, 6, 8, 12 (pontos por comprimento de onda vertical),
usando theta = 2*pi/PPWz nas formulas de m*Delta z de cada esquema.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1) Numeros de onda modificados -- os mesmos 4 esquemas de sempre
# ------------------------------------------------------------------
def mstar_dz_2a_ordem(theta):
    """Centrada, 2a ordem, nao-alternada. O(Deltaz^2)."""
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    """Centrada, 4a ordem, nao-alternada (explicita). O(Deltaz^4)."""
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    """Compacta (Pade) tridiagonal, 4a ordem (Lele, 1992). O(Deltaz^4)."""
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    """Centrada, 2a ordem, ALTERNADA (Lorenz). O(Deltaz^2)."""
    return 2.0 * np.sin(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": dict(func=mstar_dz_2a_ordem, cor="#c0392b", estilo="--"),
    "Nao-alternada, 4a ordem": dict(func=mstar_dz_4a_ordem, cor="#2980b9", estilo="-."),
    "Compacto (Pade), 4a ordem": dict(func=mstar_dz_compacto, cor="#27ae60", estilo=":"),
    "Alternada (Lorenz), 2a ordem": dict(func=mstar_dz_staggered_2a_ordem, cor="#8e44ad", estilo="-"),
}

# ------------------------------------------------------------------
# 2) PPWz avaliados e erro relativo em m* para cada esquema
# ------------------------------------------------------------------
PPWz_lista = np.array([4, 6, 8, 12])
theta_lista = 2.0 * np.pi / PPWz_lista

erros = {}
for nome, cfg in ESQUEMAS.items():
    erros[nome] = np.abs(cfg["func"](theta_lista) / theta_lista - 1.0)

# ------------------------------------------------------------------
# 3) Definicao dos tres contrastes
# ------------------------------------------------------------------
contrastes = [
    dict(
        titulo="(a) Efeito da ordem formal",
        subtitulo="não-alternada: 2ª vs 4ª ordem",
        pares=["Nao-alternada, 2a ordem", "Nao-alternada, 4a ordem"],
    ),
    dict(
        titulo="(b) Efeito da alternância (staggering)",
        subtitulo="2ª ordem: não-alternada vs alternada",
        pares=["Nao-alternada, 2a ordem", "Alternada (Lorenz), 2a ordem"],
    ),
    dict(
        titulo="(c) Efeito da compactação",
        subtitulo="4ª ordem: explícita vs compacta (Padé)",
        pares=["Nao-alternada, 4a ordem", "Compacto (Pade), 4a ordem"],
    ),
]

# ------------------------------------------------------------------
# 4) Tabelas no console -- valores e "quanto se ganha" (razao de erros)
# ------------------------------------------------------------------
print(f"{'PPWz':>6s}" + "".join(f"{nome:>28s}" for nome in ESQUEMAS))
for i, ppwz in enumerate(PPWz_lista):
    linha = f"{ppwz:6d}"
    for nome in ESQUEMAS:
        linha += f"{erros[nome][i]*100:27.3f}%"
    print(linha)

print()
for c in contrastes:
    nome_pior, nome_melhor = c["pares"]
    print(f"--- {c['titulo']} ({c['subtitulo']}) ---")
    for i, ppwz in enumerate(PPWz_lista):
        e_pior = erros[nome_pior][i]
        e_melhor = erros[nome_melhor][i]
        razao = e_pior / e_melhor if e_melhor > 0 else np.inf
        print(f"  PPWz={ppwz:3d}: {nome_pior}={e_pior*100:7.3f}%   "
              f"{nome_melhor}={e_melhor*100:7.3f}%   "
              f"-> erro {razao:6.1f}x menor")
    print()

# ------------------------------------------------------------------
# 5) Figura: 3 paineis lado a lado, um por contraste
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True)

for ax, c in zip(axes, contrastes):
    for nome in c["pares"]:
        cfg = ESQUEMAS[nome]
        ax.plot(PPWz_lista, erros[nome] * 100, cfg["estilo"], color=cfg["cor"],
                 marker="o", ms=8, lw=2.2, label=nome)
    ax.set_yscale("log")
    ax.set_xticks(PPWz_lista)
    ax.set_xlabel("PPWz")
    ax.set_title(f"{c['titulo']}\n{c['subtitulo']}", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8.5, loc="upper right")

axes[0].set_ylabel(r"Erro relativo em $m^*$ [%]  (escala log)")

fig.suptitle(
    "Experimento 3 -- Contrastes entre esquemas: ordem, alternância e compactação\n"
    "Grupo 3, MET-579",
    fontsize=13, fontweight="bold", y=1.03,
)
fig.tight_layout()
fig.savefig("experimento3_contrastes_esquemas.png", dpi=150, bbox_inches="tight")
print("Figura salva em experimento3_contrastes_esquemas.png")