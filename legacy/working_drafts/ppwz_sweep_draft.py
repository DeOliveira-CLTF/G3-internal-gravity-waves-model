# -*- coding: utf-8 -*-
"""
================================================================================
Grupo 3 - MET-579 -- VARREDURA DE PPWz (pontos por comprimento de onda vertical)
================================================================================
Extensao dos paineis (a)-(c) de dispersao_ondas_gravidade.py: em vez de fixar
so 3 resolucoes (onda de 2Deltaz, 4Deltaz, 8Deltaz), varremos continuamente o
numero de PONTOS POR COMPRIMENTO DE ONDA VERTICAL (PPWz = Lambda_z/Delta z)
de 3 a 24, igualmente espacados, mantendo:

    Delta z = 250 m   (resolucao vertical padrao, fixa)
    N       = N_BV    (Brunt-Vaisala padrao)
    f       = 0        (sem rotacao -- isola o efeito de resolucao/esquema)
    U       = 0        (sem vento de fundo -- sem deslocamento Doppler)

Relacao entre PPWz e theta (a variavel usada nas formulas de m*Delta z das
Partes 1-2): como Lambda_z = PPWz * Delta z e m = 2*pi/Lambda_z,

    theta = m * Delta z = 2*pi / PPWz

PPWz=2 seria a escala de Nyquist (2Deltaz, pior caso, theta=pi); comecamos em
PPWz=3 (theta = 2pi/3, ja' proximo do limite) e vamos ate' PPWz=24 (theta
pequeno, onda bem resolvida).

Reaproveita os mesmos 4 esquemas de diferencas finitas (mesmas formulas de
m*Delta z) das Partes 1-2. Como f=0 e U=0, a relacao de dispersao usada e' a
mesma da Parte 1 (ondas de gravidade puras); mantemos os parametros f e U
explicitos no codigo (e nao apenas omitidos) para deixar claro que essas
duas simplificacoes foram escolhas deliberadas deste experimento, e para
facilitar reativa-las depois se o grupo quiser comparar com rotacao/vento.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 0) Parametros fixos deste experimento
# ------------------------------------------------------------------
N_BV = 0.012          # frequencia de Brunt-Vaisala padrao [s^-1]
F_CORIOLIS = 0.0       # sem rotacao (isola o efeito de resolucao/esquema)
U_VENTO = 0.0          # sem vento de fundo (sem deslocamento Doppler)
dz_ref = 250.0         # Delta z fixo (resolucao vertical padrao)

# ------------------------------------------------------------------
# 1) Numeros de onda modificados (m*Delta z), identicos as Partes 1-2
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
# 2) Relacoes de dispersao (forma geral com f, reduz-se a Parte 1 se f=0)
# ------------------------------------------------------------------
def omega_analitica(k, m, N=N_BV, f=F_CORIOLIS):
    return np.sqrt((N**2 * k**2 + f**2 * m**2) / (k**2 + m**2))


def omega_numerica(k, m, dz, func_mstar, N=N_BV, f=F_CORIOLIS):
    theta = m * dz
    mstar = func_mstar(theta) / dz
    return np.sqrt((N**2 * k**2 + f**2 * mstar**2) / (k**2 + mstar**2))


# ------------------------------------------------------------------
# 3) Varredura de PPWz: 3 a 12, igualmente espacados (10 valores: 3,4,...,12)
# ------------------------------------------------------------------
PPWz_lista = np.linspace(3, 12, 10)   # 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
k_adim = np.linspace(1e-3, 4.0, 400)

# Escolhe o numero de colunas de forma a caber os paineis sem sobras (ou com
# o minimo de sobras possivel), em vez de fixar n_cols=4 -- isso evita
# paineis vazios quando a quantidade de PPWz muda.
n_paineis = len(PPWz_lista)
n_cols = 5 if n_paineis >= 5 else n_paineis
n_rows = int(np.ceil(n_paineis / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.3 * n_cols, 3.7 * n_rows), sharey=True)
axes = np.atleast_1d(axes).flatten()

print(f"Parametros fixos: Delta z = {dz_ref:.0f} m, N = {N_BV} s^-1, f = {F_CORIOLIS}, U = {U_VENTO} m/s\n")
print(f"{'PPWz':>6s}{'theta/pi':>10s}{'Lambda_z [m]':>14s}   erro relativo em m* por esquema")
for nome in ESQUEMAS:
    print(f"{'':16s}{nome}", end="")
print()

for ax, ppwz in zip(axes, PPWz_lista):
    theta0 = 2.0 * np.pi / ppwz     # PPWz = Lambda_z/Delta z = 2*pi/theta
    m = theta0 / dz_ref
    k = k_adim * m
    lambda_z = ppwz * dz_ref

    omega_a = omega_analitica(k, m) / N_BV
    ax.plot(k_adim, omega_a, color="black", lw=2.2, label="Analitica (exata)", zorder=5)

    linha_erro = f"{ppwz:6.1f}{theta0/np.pi:10.3f}{lambda_z:14.0f}   "
    for nome, cfg in ESQUEMAS.items():
        omega_n = omega_numerica(k, m, dz_ref, cfg["func"]) / N_BV
        ax.plot(k_adim, omega_n, cfg["estilo"], color=cfg["cor"], lw=1.5, label=nome)
        erro = abs(cfg["func"](theta0) / theta0 - 1.0)
        linha_erro += f"{erro*100:8.2f}%"
    print(linha_erro)

    ax.set_title(f"PPWz = {ppwz:.0f}  ($\\Lambda_z$={lambda_z:.0f} m)", fontsize=10, fontweight="bold")
    ax.set_xlabel(r"$k/m$", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)

for i in range(0, len(axes), n_cols):
    axes[i].set_ylabel(r"$\omega/N$", fontsize=9)

# Esconde posicoes de grade sobrando (caso n_paineis nao seja multiplo de n_cols)
for ax_extra in axes[n_paineis:]:
    ax_extra.set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    r"Varredura de PPWz (pontos por comprimento de onda vertical) -- $\Delta z$="
    f"{dz_ref:.0f} m fixo, f=0, U=0\nGrupo 3, MET-579",
    fontsize=13, fontweight="bold", y=1.03,
)
fig.tight_layout()
fig.savefig("varredura_ppwz.png", dpi=150, bbox_inches="tight")
print("\nFigura salva em varredura_ppwz.png")