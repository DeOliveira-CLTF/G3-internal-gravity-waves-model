# -*- coding: utf-8 -*-
"""Experimentos prognosticos que conectam dispersao e erro acumulado.

O script e independente de ``modelo_prognostico_v2.py``. Ele integra, por
RK4, o sistema linear em vorticidade-funcao de corrente para (i) um modo
normal unico e (ii) um pacote de ondas vertical. Todas as saidas sao novas e
ficam em ``resultados_experimentos_prognosticos/``.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tipografia dimensionada para figuras inseridas em pagina A4.
FONTE_BASE = 14
FONTE_EIXOS = 15
FONTE_TITULOS = 15
FONTE_TICKS = 13
FONTE_LEGENDA = 11
FONTE_PAINEIS = 16

plt.rcParams.update({
    "font.size": FONTE_BASE,
    "axes.labelsize": FONTE_EIXOS,
    "axes.titlesize": FONTE_TITULOS,
    "xtick.labelsize": FONTE_TICKS,
    "ytick.labelsize": FONTE_TICKS,
    "legend.fontsize": FONTE_LEGENDA,
    "svg.fonttype": "none",
})


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "resultados_experimentos_prognosticos"

N_BV = 0.012
LAMBDA_Z = 4000.0
M0 = 2.0 * np.pi / LAMBDA_Z
K0 = M0
PPWZ_MODO = [2, 4, 6, 8, 12, 16, 24, 32]
PPWZ_FIGURA = [4, 8, 16, 32]
PPWZ_PACOTE = [2.5, 3, 4, 8]
PERIODOS_MODO = 50.0
PERIODOS_PACOTE = 10.0
PASSOS_POR_PERIODO = 200
AMOSTRAS_MODO = 251
AMOSTRAS_PACOTE = 101


def mstar_dz_2a_ordem(theta):
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    return 2.0 * np.sin(theta / 2.0)


def derivada_mstar_2a_ordem(theta):
    return np.cos(theta)


def derivada_mstar_4a_ordem(theta):
    return (8.0 * np.cos(theta) - 2.0 * np.cos(2.0 * theta)) / 6.0


def derivada_mstar_compacto(theta):
    return 3.0 * (2.0 * np.cos(theta) + 1.0) / (2.0 + np.cos(theta)) ** 2


def derivada_mstar_staggered_2a_ordem(theta):
    return np.cos(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": {
        "func": mstar_dz_2a_ordem,
        "derivada": derivada_mstar_2a_ordem,
        "cor": "#c0392b",
        "estilo": "--",
    },
    "Nao-alternada, 4a ordem": {
        "func": mstar_dz_4a_ordem,
        "derivada": derivada_mstar_4a_ordem,
        "cor": "#2980b9",
        "estilo": "-.",
    },
    "Compacto (Pade), 4a ordem": {
        "func": mstar_dz_compacto,
        "derivada": derivada_mstar_compacto,
        "cor": "#27ae60",
        "estilo": ":",
    },
    "Alternada (Lorenz), 2a ordem": {
        "func": mstar_dz_staggered_2a_ordem,
        "derivada": derivada_mstar_staggered_2a_ordem,
        "cor": "#8e44ad",
        "estilo": "-",
    },
}


def criar_diretorio_saida():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calcular_mstar(m, dz, funcao):
    return funcao(m * dz) / dz


def omega_analitica(k, m):
    return N_BV * abs(k) / np.sqrt(k**2 + m**2)


def omega_numerica(k, m, dz, funcao):
    mstar = calcular_mstar(m, dz, funcao)
    return N_BV * abs(k) / np.sqrt(k**2 + mstar**2)


def cgz_analitica(k, m):
    return -N_BV * abs(k) * m / (k**2 + m**2) ** 1.5


def cgz_numerica(k, m, dz, funcao, derivada):
    mstar = calcular_mstar(m, dz, funcao)
    dmstar_dm = derivada(m * dz)
    return -N_BV * abs(k) * mstar * dmstar_dm / (k**2 + mstar**2) ** 1.5


def validar_finito(nome, valores):
    if not np.all(np.isfinite(valores)):
        raise ValueError(f"{nome} contem NaN ou infinito.")


def passo_rk4_modal(zeta, b, dt, k, denominador):
    """Um passo RK4 do sistema modal zeta_t=ikb; b_t=ikN^2 zeta/D."""
    def tendencia(zeta_estado, b_estado):
        return 1j * k * b_estado, 1j * k * N_BV**2 * zeta_estado / denominador

    k1z, k1b = tendencia(zeta, b)
    k2z, k2b = tendencia(zeta + 0.5 * dt * k1z, b + 0.5 * dt * k1b)
    k3z, k3b = tendencia(zeta + 0.5 * dt * k2z, b + 0.5 * dt * k2b)
    k4z, k4b = tendencia(zeta + dt * k3z, b + dt * k3b)
    zeta_novo = zeta + dt * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0
    b_novo = b + dt * (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0
    return zeta_novo, b_novo


def indices_amostragem(numero_passos, numero_amostras):
    return np.unique(np.rint(np.linspace(0, numero_passos, numero_amostras)).astype(int))


def energia_modal(psi, b, k, mstar):
    return (k**2 + mstar**2) * np.abs(psi) ** 2 + np.abs(b) ** 2 / N_BV**2


def executar_modo_normal():
    """Integra modos puros e mede frequencia, fase, erro L2 e energia."""
    omega_a = omega_analitica(K0, M0)
    periodo_a = 2.0 * np.pi / omega_a
    dt = periodo_a / PASSOS_POR_PERIODO
    numero_passos = int(round(PERIODOS_MODO * PASSOS_POR_PERIODO))
    salvar_em = set(indices_amostragem(numero_passos, AMOSTRAS_MODO).tolist())
    series = {}
    metricas = []

    for nome, cfg in ESQUEMAS.items():
        for ppwz in PPWZ_MODO:
            dz = LAMBDA_Z / ppwz
            mstar = calcular_mstar(M0, dz, cfg["func"])
            denominador = K0**2 + mstar**2
            omega_teorica = omega_numerica(K0, M0, dz, cfg["func"])

            # Mesmo k, m e amplitude de psi. A relacao entre zeta, b e psi e a
            # do autovetor de cada operador, evitando contaminar o teste com o
            # ramo de frequencia negativa.
            psi_inicial = 1.0 + 0.0j
            zeta = -denominador * psi_inicial
            b = N_BV**2 * K0 * psi_inicial / omega_teorica
            energia_inicial = float(energia_modal(psi_inicial, b, K0, mstar))

            tempos = []
            psi_numerica = []
            energia_relativa = []
            for passo in range(numero_passos + 1):
                if passo in salvar_em:
                    psi = -zeta / denominador
                    tempos.append(passo * dt)
                    psi_numerica.append(psi)
                    energia_relativa.append(float(energia_modal(psi, b, K0, mstar) / energia_inicial - 1.0))
                if passo < numero_passos:
                    zeta, b = passo_rk4_modal(zeta, b, dt, K0, denominador)

            tempos = np.asarray(tempos)
            psi_numerica = np.asarray(psi_numerica)
            energia_relativa = np.asarray(energia_relativa)
            fase_num = -np.unwrap(np.angle(psi_numerica / psi_numerica[0]))
            fase_a = omega_a * tempos
            erro_fase = fase_num - fase_a
            previsao_fase = (omega_teorica - omega_a) * tempos
            psi_a = np.exp(-1j * fase_a)
            erro_l2 = np.abs(psi_numerica - psi_a)
            omega_medida = float(np.polyfit(tempos, fase_num, 1)[0])

            for rotulo, valores in (
                ("fase", erro_fase), ("erro L2", erro_l2), ("energia", energia_relativa)
            ):
                validar_finito(f"{rotulo}, {nome}, PPWz={ppwz}", valores)

            series[(nome, ppwz)] = {
                "tempo": tempos,
                "periodos": tempos / periodo_a,
                "erro_fase": erro_fase,
                "previsao_fase": previsao_fase,
                "erro_l2": erro_l2,
                "energia_relativa": energia_relativa,
            }
            metricas.append({
                "esquema": nome,
                "PPWz": ppwz,
                "dz_m": float(dz),
                "theta_rad": float(M0 * dz),
                "omega_analitica_s-1": float(omega_a),
                "omega_numerica_teorica_s-1": float(omega_teorica),
                "omega_numerica_medida_s-1": omega_medida,
                "erro_relativo_frequencia": float((omega_medida - omega_a) / omega_a),
                "erro_fase_final_rad": float(erro_fase[-1]),
                "previsao_fase_final_rad": float(previsao_fase[-1]),
                "erro_previsao_fase_max_rad": float(np.max(np.abs(erro_fase - previsao_fase))),
                "erro_L2_final": float(erro_l2[-1]),
                "erro_L2_max": float(np.max(erro_l2)),
                "deriva_energia_max_abs": float(np.max(np.abs(energia_relativa))),
                "periodos_integrados": PERIODOS_MODO,
                "dt_s": float(dt),
            })
    return series, metricas


def centro_e_largura(z, densidade):
    peso = np.asarray(densidade, dtype=float)
    total = np.sum(peso)
    if total <= 0.0 or not np.isfinite(total):
        raise ValueError("Energia do pacote invalida.")
    centro = float(np.sum(z * peso) / total)
    largura = float(np.sqrt(np.sum((z - centro) ** 2 * peso) / total))
    return centro, largura, float(total)


def estado_pacote_exato(amplitude, omega, tempo, k, m_operador, denominador):
    psi_hat = amplitude * np.exp(-1j * omega * tempo)
    zeta_hat = -denominador * psi_hat
    b_hat = N_BV**2 * k * psi_hat / omega
    return zeta_hat, b_hat


def diagnosticar_pacote(zeta_hat, b_hat, z, k, m_operador, denominador):
    psi_hat = -zeta_hat / denominador
    psi = np.fft.ifft(psi_hat)
    b = np.fft.ifft(b_hat)
    u = np.fft.ifft(-1j * m_operador * psi_hat)
    w = np.fft.ifft(1j * k * psi_hat)
    energia = 0.5 * (np.abs(u) ** 2 + np.abs(w) ** 2 + np.abs(b) ** 2 / N_BV**2)
    centro, largura, total = centro_e_largura(z, energia)
    return centro, largura, total, psi


def executar_pacote_ondas():
    """Integra um pacote estreito em m e mede o transporte de energia."""
    omega_central = omega_analitica(K0, M0)
    periodo_central = 2.0 * np.pi / omega_central
    dt = periodo_central / PASSOS_POR_PERIODO
    numero_passos = int(round(PERIODOS_PACOTE * PASSOS_POR_PERIODO))
    salvar_em = set(indices_amostragem(numero_passos, AMOSTRAS_PACOTE).tolist())
    lz = 20.0 * LAMBDA_Z
    z0 = 0.50 * lz
    sigma_m = M0 / 20.0
    series = {}
    metricas = []

    for ppwz in PPWZ_PACOTE:
        dz = LAMBDA_Z / ppwz
        nz = int(round(lz / dz))
        z = np.arange(nz) * dz
        m = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
        janela = (m > 0.0) & (np.abs(m - M0) <= 4.0 * sigma_m)
        amplitude = np.zeros(nz, dtype=complex)
        amplitude[janela] = np.exp(-0.5 * ((m[janela] - M0) / sigma_m) ** 2) * np.exp(-1j * m[janela] * z0)

        denominador_a = K0**2 + m**2
        omega_a = N_BV * abs(K0) / np.sqrt(denominador_a)
        omega_a_segura = np.where(omega_a > 0.0, omega_a, 1.0)
        zeta_a0, b_a0 = estado_pacote_exato(amplitude, omega_a_segura, 0.0, K0, m, denominador_a)

        for nome, cfg in ESQUEMAS.items():
            mstar = calcular_mstar(m, dz, cfg["func"])
            denominador_n = K0**2 + mstar**2
            omega_n = N_BV * abs(K0) / np.sqrt(denominador_n)
            zeta_n, b_n = estado_pacote_exato(amplitude, omega_n, 0.0, K0, mstar, denominador_n)

            tempos = []
            centros_a = []
            centros_n = []
            larguras_a = []
            larguras_n = []
            energias_n = []
            perfis = {}
            for passo in range(numero_passos + 1):
                if passo in salvar_em:
                    tempo = passo * dt
                    zeta_a, b_a = estado_pacote_exato(amplitude, omega_a_segura, tempo, K0, m, denominador_a)
                    ca, la, _, _ = diagnosticar_pacote(zeta_a, b_a, z, K0, m, denominador_a)
                    cn, ln, en, psi_n = diagnosticar_pacote(zeta_n, b_n, z, K0, mstar, denominador_n)
                    tempos.append(tempo)
                    centros_a.append(ca)
                    centros_n.append(cn)
                    larguras_a.append(la)
                    larguras_n.append(ln)
                    energias_n.append(en)
                    if passo in (0, numero_passos // 2, numero_passos):
                        perfis[passo] = np.abs(psi_n) ** 2
                if passo < numero_passos:
                    zeta_n, b_n = passo_rk4_modal(zeta_n, b_n, dt, K0, denominador_n)

            tempos = np.asarray(tempos)
            centros_a = np.asarray(centros_a)
            centros_n = np.asarray(centros_n)
            larguras_a = np.asarray(larguras_a)
            larguras_n = np.asarray(larguras_n)
            energias_n = np.asarray(energias_n)
            erro_posicao = centros_n - centros_a
            cg_a_medido = float(np.polyfit(tempos, centros_a, 1)[0])
            cg_n_medido = float(np.polyfit(tempos, centros_n, 1)[0])
            cg_a_central = float(cgz_analitica(K0, M0))
            cg_n_central = float(cgz_numerica(K0, M0, dz, cfg["func"], cfg["derivada"]))
            previsao_erro = (cg_n_central - cg_a_central) * tempos

            validar_finito(f"centro do pacote, {nome}, PPWz={ppwz}", centros_n)
            validar_finito(f"erro de posicao, {nome}, PPWz={ppwz}", erro_posicao)
            series[(nome, ppwz)] = {
                "tempo": tempos,
                "periodos": tempos / periodo_central,
                "centro_analitico": centros_a,
                "centro_numerico": centros_n,
                "erro_posicao": erro_posicao,
                "previsao_erro_posicao": previsao_erro,
                "largura_analitica": larguras_a,
                "largura_numerica": larguras_n,
                "energia_relativa": energias_n / energias_n[0] - 1.0,
                "z": z,
                "perfis": perfis,
            }
            metricas.append({
                "esquema": nome,
                "PPWz": ppwz,
                "dz_m": float(dz),
                "theta_rad": float(M0 * dz),
                "cgz_analitica_central_m_s-1": cg_a_central,
                "cgz_numerica_central_m_s-1": cg_n_central,
                "cgz_analitica_medida_m_s-1": cg_a_medido,
                "cgz_numerica_medida_m_s-1": cg_n_medido,
                "erro_cgz_medido_m_s-1": cg_n_medido - cg_a_medido,
                "erro_posicao_final_m": float(erro_posicao[-1]),
                "previsao_central_erro_posicao_final_m": float(previsao_erro[-1]),
                "sigma_z_numerico_inicial_m": float(larguras_n[0]),
                "sigma_z_numerico_final_m": float(larguras_n[-1]),
                "aumento_largura_numerico_m": float(larguras_n[-1] - larguras_n[0]),
                "sigma_z_analitico_inicial_m": float(larguras_a[0]),
                "sigma_z_analitico_final_m": float(larguras_a[-1]),
                "aumento_largura_analitico_m": float(larguras_a[-1] - larguras_a[0]),
                "excesso_alargamento_numerico_m": float(
                    (larguras_n[-1] - larguras_n[0]) - (larguras_a[-1] - larguras_a[0])
                ),
                "deriva_energia_max_abs": float(np.max(np.abs(energias_n / energias_n[0] - 1.0))),
                "periodos_integrados": PERIODOS_PACOTE,
                "dt_s": float(dt),
            })
    return series, metricas


def salvar_csv(nome, linhas):
    caminho = OUTPUT_DIR / nome
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0].keys()))
        escritor.writeheader()
        escritor.writerows(linhas)


def rotular_paineis(fig, eixos):
    fig.canvas.draw()
    for letra, ax in zip("abcdefghijklmnopqrstuvwxyz", np.ravel(eixos)):
        caixa = ax.get_position()
        fig.text(caixa.x0 - 0.035, caixa.y1 + 0.012, f"({letra})", fontsize=FONTE_PAINEIS, fontweight="bold", ha="right", va="bottom")


def salvar_figura(fig, nome):
    fig.savefig(OUTPUT_DIR / f"{nome}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{nome}.svg", bbox_inches="tight")
    plt.close(fig)


def plotar_defasagem_modo(series):
    fig, eixos = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for ax, ppwz in zip(eixos.flat, PPWZ_FIGURA):
        for nome, cfg in ESQUEMAS.items():
            dados = series[(nome, ppwz)]
            ax.plot(dados["periodos"], dados["erro_fase"], cfg["estilo"], color=cfg["cor"], lw=1.8, label=nome)
            ax.plot(dados["periodos"], dados["previsao_fase"], color=cfg["cor"], lw=0.8, alpha=0.45)
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_title(rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; $\theta$={2.0 * np.pi / ppwz:.3f} rad")
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.set_ylabel(r"$\Delta\phi=\phi_{num}-\phi_{ana}$ [rad]")
        ax.grid(alpha=0.3)
    eixos.flat[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.94, wspace=0.34, hspace=0.38)
    rotular_paineis(fig, eixos)
    salvar_figura(fig, "modo_normal_defasagem")


def plotar_diagnosticos_modo(series, ppwz=8):
    fig, eixos = plt.subplots(1, 3, figsize=(16, 5.8))
    for nome, cfg in ESQUEMAS.items():
        dados = series[(nome, ppwz)]
        eixos[0].plot(dados["periodos"], dados["erro_fase"], cfg["estilo"], color=cfg["cor"], lw=1.7, label=nome)
        eixos[1].plot(dados["periodos"], dados["erro_l2"], cfg["estilo"], color=cfg["cor"], lw=1.7)
        eixos[2].semilogy(dados["periodos"], np.maximum(np.abs(dados["energia_relativa"]), 1e-18), cfg["estilo"], color=cfg["cor"], lw=1.7)
    eixos[0].set_ylabel(r"$\Delta\phi=\phi_{num}-\phi_{ana}$ [rad]")
    eixos[1].set_ylabel(r"$E_{L_2}(t)=\|\psi_{num}-\psi_{ana}\|_2/\|\psi_{ana}\|_2$")
    eixos[2].set_ylabel(r"$|\Delta E/E_0|$")
    for ax in eixos:
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.grid(alpha=0.3, which="both")
    eixos[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.18, top=0.91, wspace=0.38)
    rotular_paineis(fig, eixos)
    salvar_figura(fig, f"modo_normal_diagnosticos_PPW{ppwz}")


def plotar_centro_pacote(series):
    fig, eixos = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for ax, ppwz in zip(eixos.flat, PPWZ_PACOTE):
        dados_ref = series[(next(iter(ESQUEMAS)), ppwz)]
        ax.plot(dados_ref["periodos"], dados_ref["centro_analitico"] / 1000.0, color="black", lw=2.2, label="Analitico")
        for nome, cfg in ESQUEMAS.items():
            dados = series[(nome, ppwz)]
            ax.plot(dados["periodos"], dados["centro_numerico"] / 1000.0, cfg["estilo"], color=cfg["cor"], lw=1.8, label=nome)
        ax.set_title(rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; $\theta$={2.0 * np.pi / ppwz:.3f} rad")
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.set_ylabel(r"centro de energia $z_E$ [km]")
        ax.grid(alpha=0.3)
    eixos.flat[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.94, wspace=0.34, hspace=0.38)
    rotular_paineis(fig, eixos)
    salvar_figura(fig, "pacote_ondas_centro_energia")


def plotar_erro_posicao_pacote(series):
    fig, eixos = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for ax, ppwz in zip(eixos.flat, PPWZ_PACOTE):
        for nome, cfg in ESQUEMAS.items():
            dados = series[(nome, ppwz)]
            ax.plot(dados["periodos"], dados["erro_posicao"] / 1000.0, cfg["estilo"], color=cfg["cor"], lw=1.8, label=nome)
            ax.plot(dados["periodos"], dados["previsao_erro_posicao"] / 1000.0, color=cfg["cor"], lw=0.8, alpha=0.45)
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_title(rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; $\theta$={2.0 * np.pi / ppwz:.3f} rad")
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.set_ylabel(r"$\Delta z_E=z_{E,num}-z_{E,ana}$ [km]")
        ax.grid(alpha=0.3)
    eixos.flat[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.94, wspace=0.34, hspace=0.38)
    rotular_paineis(fig, eixos)
    salvar_figura(fig, "pacote_ondas_erro_posicao")


def plotar_alargamento_pacote(series):
    """Plota a variacao da largura energetica do pacote."""
    fig, eixos = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for ax, ppwz in zip(eixos.flat, PPWZ_PACOTE):
        dados_ref = series[(next(iter(ESQUEMAS)), ppwz)]
        delta_sigma_a = dados_ref["largura_analitica"] - dados_ref["largura_analitica"][0]
        ax.plot(
            dados_ref["periodos"],
            delta_sigma_a / 1000.0,
            color="black",
            lw=2.2,
            label="Analitico",
        )
        for nome, cfg in ESQUEMAS.items():
            dados = series[(nome, ppwz)]
            delta_sigma_n = dados["largura_numerica"] - dados["largura_numerica"][0]
            ax.plot(
                dados["periodos"],
                delta_sigma_n / 1000.0,
                cfg["estilo"],
                color=cfg["cor"],
                lw=1.8,
                label=nome,
            )
        ax.axhline(0.0, color="0.25", lw=0.8)
        ax.set_title(
            rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; "
            rf"$\theta$={2.0 * np.pi / ppwz:.3f} rad"
        )
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.set_ylabel(r"$\Delta\sigma_z=\sigma_z(t)-\sigma_z(0)$ [km]")
        ax.grid(alpha=0.3)
    eixos.flat[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(
        left=0.13, right=0.98, bottom=0.11, top=0.94,
        wspace=0.34, hspace=0.38
    )
    rotular_paineis(fig, eixos)
    salvar_figura(fig, "pacote_ondas_alargamento")

def plotar_excesso_alargamento_pacote(series):
    """Plota a deformacao numerica adicional em relacao ao pacote analitico."""
    fig, eixos = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    for ax, ppwz in zip(eixos.flat, PPWZ_PACOTE):
        for nome, cfg in ESQUEMAS.items():
            dados = series[(nome, ppwz)]
            delta_sigma_n = dados["largura_numerica"] - dados["largura_numerica"][0]
            delta_sigma_a = dados["largura_analitica"] - dados["largura_analitica"][0]
            excesso = delta_sigma_n - delta_sigma_a
            ax.plot(
                dados["periodos"],
                excesso / 1000.0,
                cfg["estilo"],
                color=cfg["cor"],
                lw=1.8,
                label=nome,
            )
        ax.axhline(0.0, color="black", lw=1.0)
        ax.set_title(
            rf"PPW$_z$={ppwz:g}; $\Delta z$={LAMBDA_Z / ppwz:.0f} m; "
            rf"$\theta$={2.0 * np.pi / ppwz:.3f} rad"
        )
        ax.set_xlabel(r"$t/T_{ana}$")
        ax.set_ylabel(
            r"$\delta\sigma_z=\Delta\sigma_{z,num}"
            r"-\Delta\sigma_{z,ana}$ [km]"
        )
        ax.grid(alpha=0.3)
    eixos.flat[0].legend(fontsize=FONTE_LEGENDA)
    fig.subplots_adjust(
        left=0.13, right=0.98, bottom=0.11, top=0.94,
        wspace=0.34, hspace=0.38
    )
    rotular_paineis(fig, eixos)
    salvar_figura(fig, "pacote_ondas_excesso_alargamento")

def escrever_resumo(metricas_modo, metricas_pacote):
    linhas = [
        "EXPERIMENTOS PROGNOSTICOS DE DISPERSAO",
        "=" * 44,
        f"N = {N_BV:.6f} s^-1",
        f"lambda_z = {LAMBDA_Z:.1f} m (fixo)",
        f"m = 2*pi/lambda_z = {M0:.10e} m^-1",
        f"k/m = {K0 / M0:.2f}",
        f"Modo normal: PPWz={PPWZ_MODO}, {PERIODOS_MODO:.0f} periodos, {PASSOS_POR_PERIODO} passos/periodo.",
        f"Pacote: PPWz={PPWZ_PACOTE}, {PERIODOS_PACOTE:.0f} periodos, dominio vertical periodico de {20 * LAMBDA_Z / 1000:.0f} km.",
        "Pacote centrado em z0=0.50*Lz, com sigma_m=m0/20 e suporte truncado em |m-m0|<=4*sigma_m.",
        "",
        "O modo normal usa um autovetor puro de cada operador, com o mesmo k, m e amplitude inicial de psi.",
        "As linhas finas nas figuras sao as previsoes espectrais; as grossas sao os resultados prognosticos RK4.",
        "O pacote usa dominio periodico para isolar dispersao de reflexao em contornos e da camada esponja.",
        "",
        "MODO NORMAL - RESULTADOS FINAIS",
    ]
    for linha in metricas_modo:
        if linha["PPWz"] in PPWZ_FIGURA:
            linhas.append(
                f"- {linha['esquema']}, PPWz={linha['PPWz']}: "
                f"erro fase={linha['erro_fase_final_rad']:.6e} rad; "
                f"L2 final={linha['erro_L2_final']:.6e}; "
                f"deriva energia max={linha['deriva_energia_max_abs']:.3e}."
            )
    linhas.extend(["", "PACOTE DE ONDAS - RESULTADOS FINAIS"])
    for linha in metricas_pacote:
        linhas.append(
            f"- {linha['esquema']}, PPWz={linha['PPWz']}: "
            f"cgz medida={linha['cgz_numerica_medida_m_s-1']:.6f} m/s; "
            f"erro posicao={linha['erro_posicao_final_m']:.3f} m; "
            f"Delta sigma_z={linha['aumento_largura_numerico_m']:.3f} m; "
            f"delta sigma_z={linha['excesso_alargamento_numerico_m']:.3f} m; "
            f"deriva energia max={linha['deriva_energia_max_abs']:.3e}."
        )
    linhas.extend([
        "",
        "Interpretacao: a defasagem nao limitada cresce aproximadamente como (omega_num-omega_ana)t.",
        "O erro L2 instantaneo e limitado e oscilatorio por causa do enrolamento da fase.",
        "O erro do centro de energia cresce aproximadamente como (cgz_num-cgz_ana)t enquanto o pacote permanece longe do contorno periodico.",
        "Deriva de energia e reportada separadamente: crescimento exponencial indicaria instabilidade, nao dispersao.",
    ])
    (OUTPUT_DIR / "resumo_experimentos_prognosticos.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def imprimir_resumo(metricas_modo, metricas_pacote):
    print("\nMODO NORMAL: frequencia e erro acumulado em 50 periodos")
    print(f"{'Esquema':35s} {'PPWz':>4s} {'Delta omega/omega_ana':>22s} {'Delta phi [rad]':>17s} {'E_L2 max':>12s} {'|dE/E0|':>12s}")
    for linha in metricas_modo:
        print(f"{linha['esquema']:35s} {linha['PPWz']:>4g} "
              f"{linha['erro_relativo_frequencia']:12.4e} {linha['erro_fase_final_rad']:17.5e} "
              f"{linha['erro_L2_max']:12.5e} {linha['deriva_energia_max_abs']:12.4e}")
    print("\nPACOTE DE ONDAS: transporte de energia em 10 periodos")
    print(f"{'Esquema':35s} {'PPWz':>4s} {'cgz [m/s]':>12s} {'Delta z_E [m]':>18s} {'Delta sigma_z [m]':>19s} {'delta sigma_z [m]':>19s} {'|dE/E0|':>12s}")
    for linha in metricas_pacote:
        print(f"{linha['esquema']:35s} {linha['PPWz']:>4g} "
              f"{linha['cgz_numerica_medida_m_s-1']:12.5f} {linha['erro_posicao_final_m']:18.3f} "
              f"{linha['aumento_largura_numerico_m']:19.3f} "
              f"{linha['excesso_alargamento_numerico_m']:19.3f} {linha['deriva_energia_max_abs']:12.4e}")


def main():
    criar_diretorio_saida()
    series_modo, metricas_modo = executar_modo_normal()
    series_pacote, metricas_pacote = executar_pacote_ondas()

    salvar_csv("metricas_modo_normal.csv", metricas_modo)
    salvar_csv("metricas_pacote_ondas.csv", metricas_pacote)
    plotar_defasagem_modo(series_modo)
    plotar_diagnosticos_modo(series_modo, ppwz=8)
    plotar_centro_pacote(series_pacote)
    plotar_erro_posicao_pacote(series_pacote)
    plotar_alargamento_pacote(series_pacote)
    plotar_excesso_alargamento_pacote(series_pacote)
    escrever_resumo(metricas_modo, metricas_pacote)
    imprimir_resumo(metricas_modo, metricas_pacote)
    print(f"\nResultados gravados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
