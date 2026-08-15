# -*- coding: utf-8 -*-
"""
================================================================================
Grupo 3 - MET-579 -- PARTE 2: efeitos fisicos adicionais para maior realismo
Tema: Impacto da resolucao vertical e do esquema de diferencas finitas na
      simulacao de ondas de gravidade internas atmosfericas.

Este script ESTENDE o modelo da Parte 1 (dispersao_ondas_gravidade.py), que
tratava apenas o caso mais idealizado (Boussinesq, sem rotacao, sem vento de
fundo). Aqui incorporamos tres efeitos fisicos discutidos no Capitulo 7 do
Holton (An Introduction to Dynamic Meteorology) para tornar o modelo mais
realista, mantendo a mesma tecnica de numero de onda vertical modificado (m*)
ja usada na Parte 1 para representar cada esquema/resolucao vertical.

Os tres efeitos adicionados sao:

  (A) ROTACAO DA TERRA -> ondas de inercia-gravidade
  (B) VENTO ZONAL MEDIO DE FUNDO -> deslocamento Doppler e nivel critico
  (C) VELOCIDADE DE GRUPO VERTICAL -> transporte de energia (analitico vs.
      numerico), grandeza tao importante quanto a frequencia para a validade
      fisica da simulacao, pois e ela (e nao a velocidade de fase) que
      transporta energia.

--------------------------------------------------------------------------------
(A) ROTACAO: ONDAS DE INERCIA-GRAVIDADE
--------------------------------------------------------------------------------
Partindo do mesmo conjunto de equacoes de Boussinesq 2D usado na Parte 1, mas
agora incluindo o termo de Coriolis (com uma componente meridional v'
acoplada por f), o sistema linearizado e:

    du'/dt - f v' = -dphi'/dx
    dv'/dt + f u'  = 0
    dw'/dt         = -dphi'/dz + b'
    db'/dt         = -N2 w'
    du'/dx + dw'/dz = 0

Resolvendo o sistema algebrico para o ansatz de onda plana exp[i(kx+mz-omegat)]
(eliminando u', v', w', b' e phi' passo a passo) chega-se a relacao de dispersao
NAO-HIDROSTATICA completa para ondas de inercia-gravidade:

                 omega2(k,m) = (N2 k2 + f2 m2) / (k2 + m2)                    (A1)

Esta expressao se reduz a duas situacoes-limite conhecidas:
  - f -> 0:            recupera a relacao de ondas de gravidade puras (Parte 1);
  - k2 << m2 (hidrostatico): omega2 -> f2 + N2k2/m2, o limite hidrostatico classico.
Ambos os limites servem como teste de consistencia da formula (A1).

Fisicamente, (A1) mostra que a frequencia de ondas de inercia-gravidade fica
sempre confinada a faixa f <= |omega| <= N (com N > f nas condicoes troposfericas
usuais) -- ou seja, a rotacao impoe um "piso" de frequencia que simplesmente
nao existe no modelo original sem rotacao. Para os periodos tipicos de
mesoescala (minutos a poucas horas), esse piso costuma ser irrelevante, mas
para ondas de periodo mais longo (dezenas de horas) a rotacao passa a
dominar a dinamica.

Versao NUMERICA: como antes, discretizamos apenas d/dz (o foco do grupo e a
resolucao vertical), substituindo m -> m* nas DUAS ocorrencias de m em (A1):

                 omega2_num = (N2 k2 + f2 m*2) / (k2 + m*2)                   (A2)


--------------------------------------------------------------------------------
(B) VENTO ZONAL MEDIO DE FUNDO: DESLOCAMENTO DOPPLER E NIVEL CRITICO
--------------------------------------------------------------------------------
Um vento basico U (constante, na direcao x) simplesmente desloca a frequencia
observada por um referencial fixo em relacao a frequencia intrinseca (a que
seria vista por um observador movendo-se com o escoamento): substituindo
d/dt -> d/dt + Ud/dx nas equacoes da Parte 1, a relacao de dispersao vira

                 omega = U k +/- omega(k,m),      omega(k,m) = N|k| / sqrt(k2+m2)       (B1)

Esse deslocamento Doppler e importante para casos realistas como ondas de
montanha (lee waves): uma onda estacionaria em relacao ao terreno (omega=0) so
existe enquanto a frequencia intrinseca |omega| = |Uk| permanecer abaixo de N;
quando |Uk| > N a onda deixa de se propagar verticalmente e passa a decair
com a altura (fica "presa"/evanescente) -- este e o chamado NIVEL CRITICO.
A versao numerica troca novamente m -> m* dentro de omega.


--------------------------------------------------------------------------------
(C) VELOCIDADE DE GRUPO VERTICAL: TRANSPORTE DE ENERGIA
--------------------------------------------------------------------------------
A energia de uma onda de gravidade se propaga na velocidade de GRUPO, nao na
de fase. Derivando omega(k,m) = N k/sqrt(k2+m2) (Parte 1) em relacao a m:

            c_gz(k,m) = domega/dm = - N k m / (k2 + m2)^(3/2)                (C1)

Um esquema numerico que representa mal m (isto e, com m* != m) tambem erra a
velocidade de grupo vertical -- e esse erro pode ser proporcionalmente maior
do que o erro na frequencia, ja que a velocidade de grupo depende da
DERIVADA da relacao de dispersao, sendo mais sensivel a erros de fase locais.
Isso e relevante porque erros na velocidade de grupo significam transporte de
energia incorreto (por exemplo, energia de ondas de montanha chegando a
estratosfera mais cedo/mais tarde ou com amplitude errada do que a fisica
real preve). Aqui calculamos c_gz numerica por diferenciacao numerica direta
da propria relacao de dispersao discreta (2), o que evita ter que derivar
analiticamente cada esquema.

USO
---
Rode `python3 dispersao_ondas_gravidade_v2_experimentos.py` para gerar figuras
separadas de rotacao, deslocamento Doppler e velocidade de grupo, alem dos
experimentos controlados. Cada figura e salva em PNG e SVG; figuras com
subplots recebem identificadores (a), (b), (c), etc., e os titulos gerais nao
sao incluidos nos arquivos exportados. Os estudantes podem alterar `LAT_GRAUS`
(latitude, define f) e `U_VENTO` (vento basico) para explorar outros regimes.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def salvar_png_svg(figura, nome_png, eixos_paineis=None):
    """Salva PNG e SVG sem titulo geral e rotula subplots, quando informados."""
    if figura._suptitle is not None:
        figura._suptitle.set_visible(False)

    if eixos_paineis is not None:
        for indice, eixo in enumerate(np.asarray(eixos_paineis).flat):
            eixo.text(
                -0.10,
                1.04,
                f"({chr(ord('a') + indice)})",
                transform=eixo.transAxes,
                fontsize=14,
                fontweight="bold",
                ha="left",
                va="bottom",
                clip_on=False,
            )

    nome_svg = nome_png.rsplit(".", 1)[0] + ".svg"
    figura.savefig(nome_png, dpi=160, bbox_inches="tight")
    figura.savefig(nome_svg, bbox_inches="tight")
    print(f"Figura salva em {nome_png} e {nome_svg}")

# ------------------------------------------------------------------
# 0) Parametros fisicos
# ------------------------------------------------------------------
N_BV = 0.012          # frequencia de Brunt-Vaisala [s^-1] (troposfera estavel)
OMEGA_TERRA = 7.292e-5  # velocidade angular da Terra [s^-1]
LAT_GRAUS = 45.0        # latitude de referencia (estudantes podem variar)
F_CORIOLIS = 2.0 * OMEGA_TERRA * np.sin(np.deg2rad(LAT_GRAUS))
U_VENTO = 15.0          # vento zonal basico de referencia [m/s] (ondas de montanha)
dz_ref = 250.0          # resolucao vertical de referencia [m]

print(f"f (Coriolis) em {LAT_GRAUS:.0f} graus = {F_CORIOLIS:.2e} s^-1  "
      f"(periodo inercial = {2*np.pi/F_CORIOLIS/3600:.1f} h)")
print(f"N/f = {N_BV/F_CORIOLIS:.1f}  (N deve ser >> f para a aproximacao ser valida)")

# ------------------------------------------------------------------
# 1) Numeros de onda modificados (m*Delta z), reaproveitados da Parte 1
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


def mstar(m, dz, func_mstar):
    """Numero de onda vertical efetivo m* enxergado pelo esquema discreto."""
    return func_mstar(m * dz) / dz


# ------------------------------------------------------------------
# (A) Ondas de inercia-gravidade: analitica e numerica -- Eqs. (A1)-(A2)
# ------------------------------------------------------------------
def omega_rotacao_analitica(k, m, f=F_CORIOLIS, N=N_BV):
    return np.sqrt((N**2 * k**2 + f**2 * m**2) / (k**2 + m**2))


def omega_rotacao_numerica(k, m, dz, func_mstar, f=F_CORIOLIS, N=N_BV):
    ms = mstar(m, dz, func_mstar)
    return np.sqrt((N**2 * k**2 + f**2 * ms**2) / (k**2 + ms**2))


# ------------------------------------------------------------------
# (B) Deslocamento Doppler por vento medio de fundo -- Eq. (B1)
# ------------------------------------------------------------------
def omega_intrinseca_analitica(k, m, N=N_BV):
    return N * np.abs(k) / np.sqrt(k**2 + m**2)


def omega_observada_analitica(k, m, U=U_VENTO, N=N_BV):
    return U * k + omega_intrinseca_analitica(k, m, N)


def omega_observada_numerica(k, m, dz, func_mstar, U=U_VENTO, N=N_BV):
    ms = mstar(m, dz, func_mstar)
    return U * k + N * np.abs(k) / np.sqrt(k**2 + ms**2)


# ------------------------------------------------------------------
# (C) Velocidade de grupo vertical -- analitica (Eq. C1) e numerica
#     (diferenciacao numerica direta da relacao de dispersao discreta)
# ------------------------------------------------------------------
def cgz_analitica(k, m, N=N_BV):
    return -N * k * m / (k**2 + m**2) ** 1.5


def cgz_numerica(k, m, dz, func_mstar, N=N_BV, dm_frac=1e-4):
    """domega_num/dm por diferenca central, aplicada diretamente sobre a
    relacao de dispersao discreta (nao sobre a formula continua)."""
    dm = dm_frac * max(abs(m), 1.0 / dz)
    om_mais = N * np.abs(k) / np.sqrt(k**2 + mstar(m + dm, dz, func_mstar) ** 2)
    om_menos = N * np.abs(k) / np.sqrt(k**2 + mstar(m - dm, dz, func_mstar) ** 2)
    return (om_mais - om_menos) / (2 * dm)


# ==================================================================
# FIGURAS SEPARADAS: uma para cada efeito fisico
# ==================================================================
figA, axA = plt.subplots(figsize=(8, 5.8))

# ---------------- Ondas de inercia-gravidade ----------------------
theta0 = np.pi / 2  # onda de 4 Delta z, resolucao intermediaria
m = theta0 / dz_ref
k_dim = np.linspace(1e-3, 4.0, 400) * m

omega_a = omega_rotacao_analitica(k_dim, m) / N_BV
axA.plot(k_dim / m, omega_a, color="black", lw=2.4, label="Analitica", zorder=5)
for nome, cfg in ESQUEMAS.items():
    om_n = omega_rotacao_numerica(k_dim, m, dz_ref, cfg["func"]) / N_BV
    axA.plot(k_dim / m, om_n, cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
axA.axhline(F_CORIOLIS / N_BV, color="gray", lw=1, ls=":", label="Piso omega=f")
axA.set_xlabel(r"$k/m$", fontsize=14)
axA.set_ylabel(r"$\omega/N$", fontsize=14)
axA.set_title("Ondas de inercia-gravidade\n(onda de 4Deltaz, f em 45 graus N)", fontsize=16, fontweight="bold")
axA.grid(alpha=0.3)
axA.legend(fontsize=10, loc="lower right")
figA.tight_layout()
salvar_png_svg(figA, "efeito_rotacao_ondas_gravidade.png")
plt.close(figA)

# ---------------- Deslocamento Doppler ----------------------------
figB, axB = plt.subplots(figsize=(8, 5.8))
# Escala de k baseada no numero de onda critico k_c = N/U (onde a frequencia
# intrinseca de uma onda estacionaria se aproxima de N); m mantido no valor
# de resolucao vertical do painel A (onda de 4 Delta z).
k_c = N_BV / U_VENTO
k_signed = np.linspace(-4.0, 4.0, 400) * k_c
om_obs_a = omega_observada_analitica(k_signed, m) / N_BV
axB.plot(k_signed / k_c, om_obs_a, color="black", lw=2.4, label="Analitica", zorder=5)
for nome, cfg in ESQUEMAS.items():
    om_obs_n = omega_observada_numerica(k_signed, m, dz_ref, cfg["func"]) / N_BV
    axB.plot(k_signed / k_c, om_obs_n, cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
axB.set_xlabel(r"$k/k_c$,  $k_c \equiv N/U$  (negativo = propagacao para oeste)", fontsize=14)
axB.set_ylabel(r"$\omega_{observada}/N$", fontsize=14)
axB.set_title(f"Deslocamento Doppler\n(vento de fundo U={U_VENTO:.0f} m/s)", fontsize=16, fontweight="bold")
axB.grid(alpha=0.3)
axB.legend(fontsize=10, loc="upper left")
figB.tight_layout()
salvar_png_svg(figB, "efeito_doppler_ondas_gravidade.png")
plt.close(figB)

# ---------------- Velocidade de grupo vertical --------------------
figC, axC = plt.subplots(figsize=(8, 5.8))
theta_range = np.linspace(0.05, np.pi * 0.98, 200)

# Mantemos a escala horizontal fixa e variamos apenas o numero de onda
# vertical. O valor de referencia de k corresponde ao m usado no painel A,
# isto e, a uma onda de 4 Delta z no ponto de referencia.
m_ref = theta0 / dz_ref
k_fixo = m_ref

# Numero de onda vertical varia ao longo do eixo horizontal
m_range = theta_range / dz_ref

cg_a = [
    cgz_analitica(k_fixo, m_var)
    for m_var in m_range
]

axC.plot(
    theta_range / np.pi,
    cg_a,
    color="black",
    lw=2.4,
    label="Analitica",
    zorder=5
)

for nome, cfg in ESQUEMAS.items():
    cg_n = [
        cgz_numerica(
            k_fixo,
            m_var,
            dz_ref,
            cfg["func"]
        )
        for m_var in m_range
    ]

    axC.plot(
        theta_range / np.pi,
        cg_n,
        cfg["estilo"],
        color=cfg["cor"],
        lw=1.7,
        label=nome
    )

axC.set_xlabel(r"$m\Delta z / \pi$")
axC.set_ylabel(r"$c_{gz}$  [m/s]")

axC.set_title(
    "Velocidade de grupo vertical\n"
    "(escala horizontal fixa)",
    fontsize=10,
    fontweight="bold"
)

axC.legend(fontsize=6.5, loc="lower left")
axC.grid(alpha=0.3)
figC.tight_layout()
salvar_png_svg(figC, "efeito_velocidade_grupo_ondas_gravidade.png")
plt.close(figC)


# ==================================================================
# EXPERIMENTOS CONTROLADOS ADICIONAIS
# ==================================================================
import csv

PPW_LIST = [4, 6, 8]
LAT_LIST = [0.0,45.0,90]

# Experimento 1: mesma onda fisica, muda apenas Delta z.
LAMBDA_Z_EXP1 = 2000.0
M_EXP1 = 2.0 * np.pi / LAMBDA_Z_EXP1
K_SOBRE_M_EXP1 = np.logspace(-3.0, np.log10(4.0), 400)

# Experimento 2
U_EXP2 = 15.0
CHI_LIST = [0.20, 0.40, 0.60, 0.75, 0.85, 0.90, 0.95]

# ==================================================================
# EXPERIMENTO 1 - RESOLUCAO VERTICAL X ROTACAO
# ==================================================================
print("\n" + "=" * 78)
print("EXPERIMENTO 1 - RESOLUCAO VERTICAL X ROTACAO")
print("=" * 78)

fig1, axes1 = plt.subplots(
    len(LAT_LIST), len(PPW_LIST),
    figsize=(17, 16), sharex=True, sharey=True
)

erro_max_exp1 = {
    nome: np.zeros((len(LAT_LIST), len(PPW_LIST)))
    for nome in ESQUEMAS
}

for i_lat, lat in enumerate(LAT_LIST):
    f_lat = 2.0 * OMEGA_TERRA * np.sin(np.deg2rad(lat))

    for j_ppw, ppw in enumerate(PPW_LIST):
        ax = axes1[i_lat, j_ppw]

        # Mesma lambda_z em todos os casos; PPW altera a grade.
        dz_exp = LAMBDA_Z_EXP1 / ppw
        k_exp = K_SOBRE_M_EXP1 * M_EXP1

        om_a = omega_rotacao_analitica(
            k_exp, M_EXP1, f=f_lat, N=N_BV
        ) / N_BV
        ax.plot(
            K_SOBRE_M_EXP1, om_a,
            color="black", lw=2.2, label="Analitica", zorder=5
        )

        for nome, cfg in ESQUEMAS.items():
            om_n = omega_rotacao_numerica(
                k_exp, M_EXP1, dz_exp, cfg["func"], f=f_lat, N=N_BV
            ) / N_BV

            ax.plot(
                K_SOBRE_M_EXP1, om_n,
                cfg["estilo"], color=cfg["cor"], lw=1.4, label=nome
            )

            erro = np.abs(om_n - om_a) / np.maximum(np.abs(om_a), 1e-12)
            erro_max_exp1[nome][i_lat, j_ppw] = 100.0 * np.max(erro)

        if f_lat > 0.0:
            ax.axhline(f_lat / N_BV, color="gray", lw=0.9, ls=":")

        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.04)
        ax.grid(alpha=0.3, which="both")
        ax.tick_params(axis="both", labelsize=14)
        if i_lat == 0:
            ax.set_title(
                f"PPWz={ppw}\nDelta z={dz_exp:.1f} m",
                fontsize=20, fontweight="bold"
            )
        if j_ppw == 0:
            ax.set_ylabel(rf"Lat={lat:.0f}$^\circ$" + "\n" + r"$\omega/N$", fontsize=16)
        if i_lat == len(LAT_LIST) - 1:
            ax.set_xlabel(r"$k/m$", fontsize=16)

handles, labels = axes1[0, 0].get_legend_handles_labels()
fig1.legend(
    handles, labels, loc="lower center", ncol=5, fontsize=16,
    bbox_to_anchor=(0.5, 0.01)
)
fig1.suptitle(
    "Experimento 1 - Interacao entre resolucao vertical e rotacao\n"
    r"Mesma onda fisica: $\lambda_z=2000$ m; $N=0.012\,s^{-1}$; $U=0$",
    fontsize=26, fontweight="bold", y=0.995
)
fig1.tight_layout(rect=[0.02, 0.05, 1.0, 0.97])
salvar_png_svg(fig1, "experimento1_rotacao_resolucao.png", axes1)
plt.close(fig1)

# Resumo quantitativo do Exp. 1
fig1b, axes1b = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
axes1b = axes1b.ravel()

for ax, (nome, cfg) in zip(axes1b, ESQUEMAS.items()):
    for i_lat, lat in enumerate(LAT_LIST):
        ax.plot(
            PPW_LIST, erro_max_exp1[nome][i_lat, :],
            marker="o", lw=1.5, label=rf"{lat:.0f}$^\circ$"
        )
    ax.set_yscale("log")
    ax.set_title(nome, fontsize=20, fontweight="bold")
    ax.set_xlabel("PPWz")
    ax.set_ylabel("Erro maximo relativo em omega [%]", fontsize=16)
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(fontsize=14)

fig1b.suptitle(
    "Experimento 1 - Erro de frequencia versus latitude e PPWz",
    fontsize=26, fontweight="bold"
)
fig1b.tight_layout(rect=[0, 0, 1, 0.95])
salvar_png_svg(fig1b, "experimento1_rotacao_erro_resumo.png", axes1b)
plt.close(fig1b)

print("Figuras do Experimento 1 salvas.")


# ==================================================================
# EXPERIMENTO 2 - APROXIMACAO DO REGIME CRITICO
# ==================================================================
print("\n" + "=" * 78)
print("EXPERIMENTO 2 - APROXIMACAO DO REGIME CRITICO")
print("=" * 78)

registros = []
erro_intrinseca = {
    nome: np.full((len(PPW_LIST), len(CHI_LIST)), np.nan)
    for nome in ESQUEMAS
}
residuo_observada = {
    nome: np.full((len(PPW_LIST), len(CHI_LIST)), np.nan)
    for nome in ESQUEMAS
}

for j_chi, chi in enumerate(CHI_LIST):
    k_exp = chi * N_BV / abs(U_EXP2)

    if chi < 1.0:
        # Onda de montanha estacionaria analitica:
        # omega_obs = U k - omega_i = 0
        m_exp = (N_BV / abs(U_EXP2)) * np.sqrt(1.0 - chi**2)
        lambda_z_exp = 2.0 * np.pi / m_exp
        omega_i_a = omega_intrinseca_analitica(k_exp, m_exp, N=N_BV)
        omega_obs_a = U_EXP2 * k_exp - omega_i_a
        regime = "propagante"

        for i_ppw, ppw in enumerate(PPW_LIST):
            dz_exp = lambda_z_exp / ppw

            for nome, cfg in ESQUEMAS.items():
                ms = mstar(m_exp, dz_exp, cfg["func"])
                omega_i_n = N_BV * abs(k_exp) / np.sqrt(k_exp**2 + ms**2)
                omega_obs_n = U_EXP2 * k_exp - omega_i_n

                erro_i = abs(omega_i_n - omega_i_a) / max(abs(omega_i_a), 1e-15)
                resid_obs = abs(omega_obs_n - omega_obs_a) / N_BV

                erro_intrinseca[nome][i_ppw, j_chi] = 100.0 * erro_i
                residuo_observada[nome][i_ppw, j_chi] = 100.0 * resid_obs

                registros.append({
                    "chi": chi,
                    "U_m_s": U_EXP2,
                    "PPWz": ppw,
                    "esquema": nome,
                    "regime_analitico": regime,
                    "k_m-1": k_exp,
                    "m_m-1": m_exp,
                    "k_sobre_m": k_exp / m_exp,
                    "lambda_z_m": lambda_z_exp,
                    "dz_m": dz_exp,
                    "omega_intrinseca_analitica_s-1": omega_i_a,
                    "omega_intrinseca_numerica_s-1": omega_i_n,
                    "omega_observada_analitica_s-1": omega_obs_a,
                    "omega_observada_numerica_s-1": omega_obs_n,
                    "erro_rel_intrinseca_pct": 100.0 * erro_i,
                    "residuo_omega_observada_sobre_N_pct": 100.0 * resid_obs,
                })
    else:
        # Para chi >= 1, m e imaginario no continuo: regime evanescente.
        for ppw in PPW_LIST:
            for nome in ESQUEMAS:
                registros.append({
                    "chi": chi, "U_m_s": U_EXP2, "PPWz": ppw,
                    "esquema": nome, "regime_analitico": "evanescente",
                    "k_m-1": k_exp, "m_m-1": np.nan, "k_sobre_m": np.nan,
                    "lambda_z_m": np.nan, "dz_m": np.nan,
                    "omega_intrinseca_analitica_s-1": chi * N_BV,
                    "omega_intrinseca_numerica_s-1": np.nan,
                    "omega_observada_analitica_s-1": np.nan,
                    "omega_observada_numerica_s-1": np.nan,
                    "erro_rel_intrinseca_pct": np.nan,
                    "residuo_omega_observada_sobre_N_pct": np.nan,
                })

# CSV
campos = list(registros[0].keys())
with open("experimento2_diagnosticos.csv", "w", newline="", encoding="utf-8") as fcsv:
    writer = csv.DictWriter(fcsv, fieldnames=campos)
    writer.writeheader()
    writer.writerows(registros)

# Figura principal do Exp. 2
PPW_LIST_FIG2 = [ppw for ppw in PPW_LIST if ppw != 12]
fig2, axes2 = plt.subplots(len(PPW_LIST_FIG2), 2, figsize=(15, 12), sharex=True)

for i_ppw, ppw in enumerate(PPW_LIST_FIG2):
    axL, axR = axes2[i_ppw, 0], axes2[i_ppw, 1]

    for nome, cfg in ESQUEMAS.items():
        axL.plot(
            CHI_LIST, erro_intrinseca[nome][i_ppw, :],
            cfg["estilo"], color=cfg["cor"],
            lw=3.6, marker="o", ms=7, label=nome
        )
        axR.plot(
            CHI_LIST, residuo_observada[nome][i_ppw, :],
            cfg["estilo"], color=cfg["cor"],
            lw=3.6, marker="o", ms=7, label=nome
        )

    axL.axvline(1.0, color="gray", ls=":", lw=1.1)
    axR.axvline(1.0, color="gray", ls=":", lw=1.1)
    axL.set_yscale("log")
    axR.set_yscale("log")
    axL.set_ylabel(f"PPWz={ppw}\nErro $\\omega_i$ [%]", fontsize=16)
    axR.set_ylabel(f"PPWz={ppw}\n" + r"$|\omega_{obs,num}|/N$ [%]", fontsize=16)
    axL.grid(alpha=0.3, which="both")
    axR.grid(alpha=0.3, which="both")
    axL.tick_params(axis="both", labelsize=14)
    axR.tick_params(axis="both", labelsize=14)

    if i_ppw == 0:
        axL.set_title("Erro relativo da frequencia intrinseca",
                      fontsize=20, fontweight="bold")
        axR.set_title("Residual de estacionariedade",
                      fontsize=20, fontweight="bold")

axes2[-1, 0].set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)
axes2[-1, 1].set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)

handles, labels = axes2[0, 0].get_legend_handles_labels()
fig2.legend(handles, labels, loc="lower center", ncol=4, fontsize=16,
            bbox_to_anchor=(0.5, 0.01))
fig2.suptitle(
    "Sensibilidade da frequência intrínseca e da estacionariedade à discretização vertical",
    fontsize=26, fontweight="bold"
)
fig2.tight_layout(rect=[0.02, 0.05, 1.0, 0.94])
salvar_png_svg(fig2, "experimento2_regime_critico.png", axes2)
plt.close(fig2)

# Heatmaps
fig2b, axes2b = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
axes2b = axes2b.ravel()

for ax, (nome, cfg) in zip(axes2b, ESQUEMAS.items()):
    mesh = ax.pcolormesh(CHI_LIST, PPW_LIST, erro_intrinseca[nome], shading="auto")
    fig2b.colorbar(mesh, ax=ax, label="Erro $\\omega_i$ [%]")
    ax.set_title(nome, fontsize=20, fontweight="bold")
    ax.set_xlabel(r"$\chi=|Uk|/N$", fontsize=16)
    ax.set_ylabel("PPWz", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)

fig2b.suptitle(
    "Experimento 2 - Mapa do erro da frequencia intrinseca",
    fontsize=26, fontweight="bold"
)
fig2b.tight_layout(rect=[0, 0, 1, 0.95])
salvar_png_svg(fig2b, "experimento2_regime_critico_heatmaps.png", axes2b)
plt.close(fig2b)

# ==================================================================
# EXPERIMENTO DE ROTACAO REFORMULADO
# Fisica: importancia relativa de f/N
# Numerica: erro de frequencia em funcao de (k/m, f/N)
# ==================================================================

print("\n" + "=" * 78)
print("EXPERIMENTO ROTACAO - EFEITO DE f/N E RESOLUCAO VERTICAL")
print("=" * 78)

# ------------------------------------------------------------------
# Parametros do experimento
# ------------------------------------------------------------------

# Valores de latitude para referencia fisica
LAT_ROT = [0.0, 30.0, 60.0, 90.0]

# Valores de N para explorar diferentes intensidades de estratificacao
N_ROT = [0.003, 0.006, 0.012, 0.018]

# PPW usados na parte numerica
PPW_ROT = [4, 6, 8]

# Mantemos a MESMA onda vertical fisica
LAMBDA_Z_ROT = 2000.0
M_ROT = 2.0 * np.pi / LAMBDA_Z_ROT

# Faixa focada na regiao onde Coriolis importa:
# k/m muito pequeno
K_SOBRE_M_FISICA = np.logspace(-4.0, -0.3, 400)

# Faixa um pouco mais ampla para o mapa numerico
K_SOBRE_M_MAPA = np.logspace(-4.0, 0.3, 220)

# Grade continua de f/N para os mapas
F_SOBRE_N_MAPA = np.logspace(-4.0, -0.2, 180)


# ==================================================================
# PARTE 1 - FIGURA FISICA
# omega/N versus k/m para diferentes f/N
# ==================================================================

fig_rot1, ax_rot1 = plt.subplots(figsize=(10, 7))

# Vamos usar N=0.012 apenas como referencia para converter latitude em f/N.
N_REF_ROT = 0.012

for lat in LAT_ROT:
    f_lat = 2.0 * OMEGA_TERRA * np.sin(np.deg2rad(lat))
    f_sobre_N = f_lat / N_REF_ROT

    k_rot = K_SOBRE_M_FISICA * M_ROT

    omega_rot = omega_rotacao_analitica(
        k_rot,
        M_ROT,
        f=f_lat,
        N=N_REF_ROT
    ) / N_REF_ROT

    ax_rot1.plot(
        K_SOBRE_M_FISICA,
        omega_rot,
        lw=2.4,
        label=rf"{lat:.0f}$^\circ$  ($f/N={f_sobre_N:.3f}$)"
    )

ax_rot1.set_xscale("log")
ax_rot1.set_xlabel(r"$k/m$", fontsize=16)
ax_rot1.set_ylabel(r"$\omega/N$", fontsize=16)
ax_rot1.set_title(
    "Efeito da rotacao sobre a relacao de dispersao\n"
    r"($N=0.012\ s^{-1}$; mesma onda vertical)",
    fontsize=20,
    fontweight="bold"
)
ax_rot1.grid(alpha=0.3, which="both")
ax_rot1.tick_params(axis="both", labelsize=14)
ax_rot1.legend(fontsize=12)

fig_rot1.tight_layout()
salvar_png_svg(fig_rot1, "rotacao_fisica_latitude.png")
plt.close(fig_rot1)

print("[OK] rotacao_fisica_latitude.png")


# ==================================================================
# PARTE 1B - FIGURA FISICA VARIANDO N E LATITUDE
# Mostra diretamente como f/N muda
# ==================================================================

fig_rot1b, axes_rot1b = plt.subplots(
    len(N_ROT),
    1,
    figsize=(10, 13),
    sharex=True
)

for iN, N_val in enumerate(N_ROT):
    ax = axes_rot1b[iN]

    for lat in LAT_ROT:
        f_lat = 2.0 * OMEGA_TERRA * np.sin(np.deg2rad(lat))
        f_sobre_N = f_lat / N_val

        k_rot = K_SOBRE_M_FISICA * M_ROT

        omega_rot = omega_rotacao_analitica(
            k_rot,
            M_ROT,
            f=f_lat,
            N=N_val
        ) / N_val

        ax.plot(
            K_SOBRE_M_FISICA,
            omega_rot,
            lw=2.2,
            label=rf"{lat:.0f}$^\circ$"
        )

    ax.set_xscale("log")
    ax.set_ylabel(r"$\omega/N$", fontsize=14)
    ax.set_title(
        rf"$N={N_val:.3f}\ s^{{-1}}$",
        fontsize=16,
        fontweight="bold"
    )
    ax.grid(alpha=0.3, which="both")
    ax.tick_params(axis="both", labelsize=12)

axes_rot1b[-1].set_xlabel(r"$k/m$", fontsize=16)

handles, labels = axes_rot1b[0].get_legend_handles_labels()
fig_rot1b.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    fontsize=12,
    bbox_to_anchor=(0.5, 0.01)
)

fig_rot1b.suptitle(
    "Competicao entre rotacao e estratificacao\n"
    r"Efeito de latitude e $N$ sobre $\omega/N$",
    fontsize=20,
    fontweight="bold"
)

fig_rot1b.tight_layout(rect=[0, 0.05, 1, 0.95])
salvar_png_svg(fig_rot1b, "rotacao_fisica_latitude_N.png", axes_rot1b)
plt.close(fig_rot1b)

print("[OK] rotacao_fisica_latitude_N.png")


# ==================================================================
# PARTE 2 - MAPAS DE ERRO NUMERICO
# E_omega em funcao de (k/m, f/N)
#
# IMPORTANTE:
# - mesma escala de cores para TODOS os PPWz
# - maximo global calculado considerando:
#       todos os PPWz
#       todos os esquemas
#       todos os valores de k/m e f/N
# - colorbar em eixo proprio, sem sobrepor os subplots
# - espaco superior reservado para o supertitulo
# ==================================================================

print("\nCalculando escala global de erro para os mapas de rotacao...")

# ------------------------------------------------------------------
# Grade 2D comum a todos os PPWz
# ------------------------------------------------------------------
KM_GRID, FN_GRID = np.meshgrid(
    K_SOBRE_M_MAPA,
    F_SOBRE_N_MAPA
)

# Como a relacao esta adimensionalizada por N,
# podemos usar N=1 nesta parte.
N_adim = 1.0

# f = (f/N)*N
F_GRID = FN_GRID * N_adim

# k = (k/m)*m
K_GRID = KM_GRID * M_ROT

# Frequencia analitica: igual para todos os PPWz,
# pois depende da onda fisica e nao da discretizacao.
OMEGA_A_GLOBAL = np.sqrt(
    (
        N_adim**2 * K_GRID**2
        +
        F_GRID**2 * M_ROT**2
    )
    /
    (
        K_GRID**2 + M_ROT**2
    )
)

# ------------------------------------------------------------------
# PRIMEIRA PASSAGEM:
# calcula TODOS os erros antes de plotar.
#
# Estrutura:
# erros_rotacao[ppw][nome_esquema] = matriz 2D de erro
# ------------------------------------------------------------------
erros_rotacao = {}

vmax_global_rotacao = 0.0

for ppw in PPW_ROT:

    dz_rot = LAMBDA_Z_ROT / ppw

    erros_rotacao[ppw] = {}

    for nome, cfg in ESQUEMAS.items():

        ms = mstar(
            M_ROT,
            dz_rot,
            cfg["func"]
        )

        OMEGA_N = np.sqrt(
            (
                N_adim**2 * K_GRID**2
                +
                F_GRID**2 * ms**2
            )
            /
            (
                K_GRID**2 + ms**2
            )
        )

        ERRO = (
            np.abs(
                OMEGA_N - OMEGA_A_GLOBAL
            )
            /
            np.maximum(
                np.abs(OMEGA_A_GLOBAL),
                1e-14
            )
        ) * 100.0

        erros_rotacao[ppw][nome] = ERRO

        # Atualiza maximo considerando TODOS os PPWz e esquemas
        vmax_global_rotacao = max(
            vmax_global_rotacao,
            np.nanmax(ERRO)
        )

print(
    f"Maximo global de erro usado em todas as colorbars: "
    f"{vmax_global_rotacao:.2f}%"
)


# ------------------------------------------------------------------
# SEGUNDA PASSAGEM:
# agora desenhamos os mapas usando EXATAMENTE o mesmo
# vmin e vmax em todos os PPWz.
# ------------------------------------------------------------------
for ppw in PPW_ROT:

    # --------------------------------------------------------------
    # Criamos a figura com GridSpec.
    #
    # As duas primeiras colunas sao os subplots.
    # A terceira coluna e EXCLUSIVA para a colorbar.
    #
    # Isso impede qualquer sobreposicao.
    # --------------------------------------------------------------
    fig_rot2 = plt.figure(
        figsize=(15, 11)
    )

    gs = fig_rot2.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.0, 1.0, 0.055],
        height_ratios=[1.0, 1.0],
        left=0.075,
        right=0.93,
        bottom=0.085,
        top=0.82,
        wspace=0.28,
        hspace=0.34
    )

    # Quatro paineis
    ax00 = fig_rot2.add_subplot(gs[0, 0])
    ax01 = fig_rot2.add_subplot(
        gs[0, 1],
        sharex=ax00,
        sharey=ax00
    )
    ax10 = fig_rot2.add_subplot(
        gs[1, 0],
        sharex=ax00,
        sharey=ax00
    )
    ax11 = fig_rot2.add_subplot(
        gs[1, 1],
        sharex=ax00,
        sharey=ax00
    )

    axes_rot2 = [
        ax00,
        ax01,
        ax10,
        ax11
    ]

    # Eixo proprio e exclusivo da colorbar
    cax = fig_rot2.add_subplot(
        gs[:, 2]
    )

    mesh = None

    for ax, (nome, cfg) in zip(
        axes_rot2,
        ESQUEMAS.items()
    ):

        ERRO = erros_rotacao[ppw][nome]

        mesh = ax.pcolormesh(
            K_SOBRE_M_MAPA,
            F_SOBRE_N_MAPA,
            ERRO,
            shading="auto",
            rasterized=True,

            # MESMA escala para todos os PPWs
            vmin=0.0,
            vmax=vmax_global_rotacao
        )

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(
            nome,
            fontsize=17,
            fontweight="bold",
            pad=10
        )

        ax.set_xlabel(
            r"$k/m$",
            fontsize=15
        )

        ax.set_ylabel(
            r"$f/N$",
            fontsize=15
        )

        ax.tick_params(
            axis="both",
            labelsize=12
        )

        ax.grid(
            alpha=0.15,
            which="both"
        )

    # --------------------------------------------------------------
    # Colorbar unica, agora em eixo separado
    # --------------------------------------------------------------
    cbar = fig_rot2.colorbar(
        mesh,
        cax=cax
    )

    cbar.set_label(
        r"Erro relativo de $\omega$ [%]",
        fontsize=15,
        labelpad=14
    )

    cbar.ax.tick_params(
        labelsize=12
    )

    # --------------------------------------------------------------
    # Supertitulo:
    #
    # y=0.955 deixa espaco grande em relacao aos titulos dos paineis.
    # O top do GridSpec esta em 0.82, logo ha uma faixa exclusiva
    # para o titulo.
    # --------------------------------------------------------------
    fig_rot2.suptitle(
        r"Erro numérico da frequência em função de $k/m$ e $f/N$"
        "\n"
        rf"PPW$_z$ = {ppw}; "
        rf"mesma onda vertical "
        rf"$\lambda_z={LAMBDA_Z_ROT:.0f}$ m",
        fontsize=22,
        fontweight="bold",
        y=0.965
    )

    nome_saida = (
        f"rotacao_erro_km_fN_PPW{ppw}.png"
    )

    salvar_png_svg(fig_rot2, nome_saida, axes_rot2)

    plt.close(fig_rot2)

    print(
        f"[OK] {nome_saida} "
        f"(escala comum: 0 a "
        f"{vmax_global_rotacao:.2f}%)"
    )

print("Figuras e CSV do Experimento 2 salvos.")
