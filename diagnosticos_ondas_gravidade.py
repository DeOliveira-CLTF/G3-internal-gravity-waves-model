# -*- coding: utf-8 -*-
"""Diagnosticos quantitativos adicionais para ondas de gravidade internas.

Este script e independente dos codigos originais: reproduz somente as formulas
matematicas necessarias e grava todas as novas saidas em
``resultados_diagnosticos/``.
"""

import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "resultados_diagnosticos"

N_BV = 0.012
LAMBDA_Z = 4000.0
M = 2.0 * np.pi / LAMBDA_Z
K_SOBRE_M = np.linspace(0.05, 4.0, 400)
K_SOBRE_M_CG = np.linspace(0.05, 4.0, 400)
PONTOS_POR_ONDA = [2, 4, 6, 8, 12, 16, 24, 32]
PONTOS_CONVERGENCIA = [8, 12, 16, 24, 32]
PONTOS_CG = [4, 8, 16, 32]


def criar_diretorio_saida():
    """Cria e retorna o diretorio exclusivo das novas saidas."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def mstar_dz_2a_ordem(theta):
    return np.sin(theta)


def mstar_dz_4a_ordem(theta):
    return (8.0 * np.sin(theta) - np.sin(2.0 * theta)) / 6.0


def mstar_dz_compacto(theta):
    return 3.0 * np.sin(theta) / (2.0 + np.cos(theta))


def mstar_dz_staggered_2a_ordem(theta):
    return 2.0 * np.sin(theta / 2.0)


ESQUEMAS = {
    "Nao-alternada, 2a ordem": dict(func=mstar_dz_2a_ordem, cor="#c0392b", estilo="--", ordem=2.0),
    "Nao-alternada, 4a ordem": dict(func=mstar_dz_4a_ordem, cor="#2980b9", estilo="-.", ordem=4.0),
    "Compacto (Pade), 4a ordem": dict(func=mstar_dz_compacto, cor="#27ae60", estilo=":", ordem=4.0),
    "Alternada (Lorenz), 2a ordem": dict(func=mstar_dz_staggered_2a_ordem, cor="#8e44ad", estilo="-", ordem=2.0),
}


def avisar(mensagem):
    print(f"AVISO: {mensagem}")


def validar_finito(nome, valores):
    if not np.all(np.isfinite(valores)):
        raise ValueError(f"{nome} contem NaN ou infinito.")


def calcular_mstar(m, dz, funcao_mstar):
    return funcao_mstar(m * dz) / dz


def omega_analitica(k, m, N=N_BV):
    return N * np.abs(k) / np.sqrt(k**2 + m**2)


def omega_numerica(k, m, dz, funcao_mstar, N=N_BV):
    mstar = calcular_mstar(m, dz, funcao_mstar)
    denominador = np.sqrt(k**2 + mstar**2)
    return np.divide(N * np.abs(k), denominador, out=np.zeros_like(k, dtype=float), where=denominador > 0.0)


def validar_metricas(nome, l2, linf):
    if l2 < 0.0 or linf < 0.0:
        avisar(f"metrica negativa em {nome}: L2={l2}, Linf={linf}.")
    if linf + 10.0 * np.finfo(float).eps < l2:
        avisar(f"Linf < L2 em {nome}: L2={l2}, Linf={linf}.")


def calcular_metricas_frequencia():
    linhas = []
    k = K_SOBRE_M * M
    omega_a = omega_analitica(k, M)
    validar_finito("omega analitica", omega_a)

    for nome, cfg in ESQUEMAS.items():
        for pontos in PONTOS_POR_ONDA:
            dz = LAMBDA_Z / pontos
            theta = M * dz
            mstar = calcular_mstar(M, dz, cfg["func"])
            omega_n = omega_numerica(k, M, dz, cfg["func"])
            delta = (omega_n - omega_a) / N_BV
            validar_finito(f"delta_omega ({nome}, P={pontos})", delta)
            l2 = float(np.sqrt(np.mean(delta**2)))
            linf = float(np.max(np.abs(delta)))
            bias = float(np.mean(delta))
            erro_mstar = float(abs(mstar / M - 1.0))
            validar_metricas(f"frequencia ({nome}, P={pontos})", l2, linf)
            linhas.append({
                "esquema": nome,
                "pontos_por_comprimento_de_onda": pontos,
                "dz_m": float(dz),
                "theta": float(theta),
                "erro_mstar": erro_mstar,
                "L2_omega": l2,
                "Linf_omega": linf,
                "bias_omega": bias,
            })
    return linhas


def estimar_ordens_convergencia(metricas):
    globais = []
    pares = []
    for nome, cfg in ESQUEMAS.items():
        dados = [linha for linha in metricas if linha["esquema"] == nome and
                 linha["pontos_por_comprimento_de_onda"] in PONTOS_CONVERGENCIA]
        dados.sort(key=lambda linha: PONTOS_CONVERGENCIA.index(linha["pontos_por_comprimento_de_onda"]))
        dz = np.array([linha["dz_m"] for linha in dados])
        erros = np.array([linha["L2_omega"] for linha in dados])
        validar_finito(f"erros de convergencia ({nome})", erros)
        if np.any(erros <= 0.0):
            raise ValueError(f"Nao e possivel ajustar log de erro nao positivo para {nome}.")
        ordem_global = float(np.polyfit(np.log(dz), np.log(erros), 1)[0])
        globais.append({
            "esquema": nome,
            "ordem_global": ordem_global,
            "pontos_inicio": PONTOS_CONVERGENCIA[0],
            "pontos_fim": PONTOS_CONVERGENCIA[-1],
        })
        if abs(ordem_global - cfg["ordem"]) > 1.0:
            avisar(f"ordem observada de {nome} ({ordem_global:.3f}) distante de {cfg['ordem']:.0f}.")
        for anterior, atual in zip(dados[:-1], dados[1:]):
            ordem_par = float(
                np.log(anterior["L2_omega"] / atual["L2_omega"])
                / np.log(anterior["dz_m"] / atual["dz_m"])
            )
            pares.append({
                "esquema": nome,
                "pontos_inicio": anterior["pontos_por_comprimento_de_onda"],
                "pontos_fim": atual["pontos_por_comprimento_de_onda"],
                "ordem_entre_pares": ordem_par,
            })
        if not np.all(np.diff(erros) < 0.0):
            avisar(f"o erro de frequencia de {nome} nao diminui estritamente para P >= 8.")
    return globais, pares


def cgz_analitica(k, m, N=N_BV):
    return -N * k * m / (k**2 + m**2) ** 1.5


def cgz_numerica(k, m, dz, funcao_mstar, N=N_BV):
    dm = 1.0e-4 * max(abs(m), 1.0 / dz)
    mstar_mais = calcular_mstar(m + dm, dz, funcao_mstar)
    mstar_menos = calcular_mstar(m - dm, dz, funcao_mstar)
    omega_mais = N * np.abs(k) / np.sqrt(k**2 + mstar_mais**2)
    omega_menos = N * np.abs(k) / np.sqrt(k**2 + mstar_menos**2)
    return (omega_mais - omega_menos) / (2.0 * dm)


def calcular_metricas_cgz():
    linhas = []
    k = K_SOBRE_M_CG * M
    cg_a = cgz_analitica(k, M)
    denominador = float(np.max(np.abs(cg_a)))
    if denominador <= 0.0 or not np.isfinite(denominador):
        raise ValueError("Denominador global invalido para o erro de velocidade de grupo.")

    for nome, cfg in ESQUEMAS.items():
        erros_por_refinamento = []
        for pontos in PONTOS_CG:
            dz = LAMBDA_Z / pontos
            cg_n = cgz_numerica(k, M, dz, cfg["func"])
            erro = (cg_n - cg_a) / denominador
            validar_finito(f"erro_cgz ({nome}, P={pontos})", erro)
            l2 = float(np.sqrt(np.mean(erro**2)))
            linf = float(np.max(np.abs(erro)))
            bias = float(np.mean(erro))
            validar_metricas(f"velocidade de grupo ({nome}, P={pontos})", l2, linf)
            erros_por_refinamento.append(l2)
            linhas.append({
                "esquema": nome,
                "pontos_por_comprimento_de_onda": pontos,
                "dz_m": float(dz),
                "L2_cgz": l2,
                "Linf_cgz": linf,
                "bias_cgz": bias,
            })
        if not np.all(np.diff(erros_por_refinamento) < 0.0):
            avisar(f"o erro de velocidade de grupo de {nome} nao diminui estritamente em {PONTOS_CG}.")
    return linhas


def salvar_csv(caminho, linhas, colunas):
    with caminho.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=colunas)
        escritor.writeheader()
        escritor.writerows(linhas)


def plotar_erro_frequencia():
    k = K_SOBRE_M * M
    omega_a = omega_analitica(k, M)
    fig, eixos = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, pontos in zip(eixos.flat, [2, 4, 8, 16]):
        dz = LAMBDA_Z / pontos
        theta = M * dz
        for nome, cfg in ESQUEMAS.items():
            erro = np.abs((omega_numerica(k, M, dz, cfg["func"]) - omega_a) / N_BV)
            validar_finito(f"figura de erro ({nome}, P={pontos})", erro)
            ax.semilogy(K_SOBRE_M, np.maximum(erro, np.finfo(float).tiny), cfg["estilo"],
                        color=cfg["cor"], lw=1.8, label=nome)
        ax.set_title(rf"P={pontos}; $\Delta z$={dz:.1f} m; $\theta$={theta:.4f} rad")
        ax.set_xlabel(r"$k/m$ (adimensional)")
        ax.set_ylabel(r"$|\omega_{num}-\omega_{ana}|/N$ (adimensional)")
        ax.grid(alpha=0.3, which="both")
    eixos.flat[0].legend(fontsize=8)
    fig.suptitle("Erro normalizado da frequencia no espaco espectral", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "erro_frequencia_por_k.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_convergencia_frequencia(metricas, ordens_globais):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    mapa_ordens = {linha["esquema"]: linha["ordem_global"] for linha in ordens_globais}
    todos_erros = []
    for nome, cfg in ESQUEMAS.items():
        dados = [linha for linha in metricas if linha["esquema"] == nome]
        dados.sort(key=lambda linha: linha["dz_m"], reverse=True)
        dz = np.array([linha["dz_m"] for linha in dados])
        erros = np.array([linha["L2_omega"] for linha in dados])
        todos_erros.extend(erros)
        ax.loglog(dz, erros, cfg["estilo"], color=cfg["cor"], marker="o", lw=1.8,
                  label=f"{nome} — p={mapa_ordens[nome]:.2f}")
    dz_ref = np.array([LAMBDA_Z / 8.0, LAMBDA_Z / 32.0])
    escala = float(np.median(todos_erros))
    ax.loglog(dz_ref, escala * (dz_ref / dz_ref[0]) ** 2, color="0.35", ls="--", label=r"$O(\Delta z^2)$")
    ax.loglog(dz_ref, escala * 0.25 * (dz_ref / dz_ref[0]) ** 4, color="0.35", ls=":", label=r"$O(\Delta z^4)$")
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta z$ [m] (refinamento para a direita)")
    ax.set_ylabel(r"$L_2$ do erro normalizado de frequencia")
    ax.set_title("Convergencia da frequencia", fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergencia_frequencia.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_velocidade_grupo():
    pontos = 8
    dz = LAMBDA_Z / pontos
    k = K_SOBRE_M_CG * M
    cg_a = cgz_analitica(k, M)
    denominador = float(np.max(np.abs(cg_a)))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    ax1.plot(K_SOBRE_M_CG, cg_a, color="black", lw=2.4, label="Analitica", zorder=5)
    for nome, cfg in ESQUEMAS.items():
        cg_n = cgz_numerica(k, M, dz, cfg["func"])
        erro = np.abs(cg_n - cg_a) / denominador
        ax1.plot(K_SOBRE_M_CG, cg_n, cfg["estilo"], color=cfg["cor"], lw=1.8, label=nome)
        ax2.semilogy(K_SOBRE_M_CG, np.maximum(erro, np.finfo(float).tiny), cfg["estilo"],
                     color=cfg["cor"], lw=1.8, label=nome)
    ax1.set_ylabel(r"$c_{gz}$ [m s$^{-1}$]")
    ax1.set_title(rf"Velocidade de grupo vertical — P={pontos}, $\Delta z$={dz:.1f} m", fontweight="bold")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)
    ax2.set_xlabel(r"$k/m$ (adimensional)")
    ax2.set_ylabel(r"$|c_{gz,num}-c_{gz,ana}|/\max|c_{gz,ana}|$")
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "velocidade_grupo_e_erro.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plotar_convergencia_cgz(metricas_cgz):
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for nome, cfg in ESQUEMAS.items():
        dados = [linha for linha in metricas_cgz if linha["esquema"] == nome]
        dados.sort(key=lambda linha: linha["dz_m"], reverse=True)
        ax.loglog([linha["dz_m"] for linha in dados], [linha["L2_cgz"] for linha in dados],
                  cfg["estilo"], color=cfg["cor"], marker="o", lw=1.8, label=nome)
    ax.invert_xaxis()
    ax.set_xlabel(r"$\Delta z$ [m] (refinamento para a direita)")
    ax.set_ylabel(r"$L_2$ do erro normalizado de $c_{gz}$")
    ax.set_title("Convergencia da velocidade de grupo vertical", fontweight="bold")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "convergencia_velocidade_grupo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def escrever_resumo(metricas_frequencia, ordens_globais, metricas_cgz):
    pontos_comuns = max(PONTOS_POR_ONDA)
    freq_fina = [linha for linha in metricas_frequencia
                 if linha["pontos_por_comprimento_de_onda"] == pontos_comuns]
    cg_fina = [linha for linha in metricas_cgz
               if linha["pontos_por_comprimento_de_onda"] == max(PONTOS_CG)]
    melhor_l2 = min(freq_fina, key=lambda linha: linha["L2_omega"])
    melhor_linf = min(freq_fina, key=lambda linha: linha["Linf_omega"])
    melhor_cg = min(cg_fina, key=lambda linha: linha["L2_cgz"])

    linhas = [
        "RESUMO DOS DIAGNOSTICOS DE ONDAS DE GRAVIDADE",
        "=" * 52,
        f"N_BV = {N_BV:.6g} s^-1",
        f"lambda_z = {LAMBDA_Z:.1f} m",
        f"m = 2*pi/lambda_z = {M:.10e} m^-1",
        "Intervalo k/m = 0.05 a 4.0 (400 pontos).",
        "",
        "O comprimento de onda vertical fisico lambda_z e o numero de onda M foram mantidos fixos; apenas dz variou.",
        f"Resolucoes verticais de frequencia (pontos por onda): {PONTOS_POR_ONDA}",
        f"Resolucoes verticais de velocidade de grupo: {PONTOS_CG}",
        "",
        "Ordens globais observadas (ajuste com P >= 8):",
    ]
    linhas.extend(f"- {linha['esquema']}: p = {linha['ordem_global']:.6f}"
                  for linha in ordens_globais)
    linhas.extend([
        "",
        f"Comparacao dos esquemas na resolucao comum mais fina, P={pontos_comuns}:",
        f"- Menor L2 de frequencia: {melhor_l2['esquema']} ({melhor_l2['L2_omega']:.10e})",
        f"- Menor Linf de frequencia: {melhor_linf['esquema']} ({melhor_linf['Linf_omega']:.10e})",
        f"- Menor L2 da velocidade de grupo: {melhor_cg['esquema']} ({melhor_cg['L2_cgz']:.10e})",
        "",
        "P=2 representa a escala de Nyquist. Nesse limite, mstar pode ser nulo nos esquemas nao alternados, sem causar divisao por zero neste experimento.",
        "A ordem de convergencia foi estimada somente com P >= 8, evitando o regime mal resolvido.",
        "Os resultados sao diagnosticos numericos deste conjunto de parametros e nao implicam conclusoes fisicas gerais.",
    ])
    (OUTPUT_DIR / "resumo_diagnosticos.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")


def imprimir_tabela_frequencia(linhas):
    print("\nMETRICAS DE FREQUENCIA")
    cabecalho = (f"{'Esquema':35s} {'P':>3s} {'dz [m]':>10s} {'theta':>10s} "
                 f"{'erro m*':>12s} {'L2':>12s} {'Linf':>12s} {'bias':>12s}")
    print(cabecalho)
    print("-" * len(cabecalho))
    for linha in linhas:
        print(f"{linha['esquema']:35s} {linha['pontos_por_comprimento_de_onda']:3d} "
              f"{linha['dz_m']:10.3f} {linha['theta']:10.6f} {linha['erro_mstar']:12.5e} "
              f"{linha['L2_omega']:12.5e} {linha['Linf_omega']:12.5e} {linha['bias_omega']:12.5e}")


def imprimir_resumo_ordens_cgz(ordens, metricas_cgz):
    print("\nORDENS GLOBAIS OBSERVADAS (P >= 8)")
    for linha in ordens:
        print(f"  {linha['esquema']:35s} p = {linha['ordem_global']:.6f}")
    print("\nMETRICAS DE VELOCIDADE DE GRUPO")
    print(f"{'Esquema':35s} {'P':>3s} {'dz [m]':>10s} {'L2_cgz':>12s} {'Linf_cgz':>12s} {'bias_cgz':>12s}")
    for linha in metricas_cgz:
        print(f"{linha['esquema']:35s} {linha['pontos_por_comprimento_de_onda']:3d} "
              f"{linha['dz_m']:10.3f} {linha['L2_cgz']:12.5e} "
              f"{linha['Linf_cgz']:12.5e} {linha['bias_cgz']:12.5e}")


def validar_limite_ondas_longas():
    theta = 1.0e-6
    for nome, cfg in ESQUEMAS.items():
        razao = float(cfg["func"](theta) / theta)
        if not np.isfinite(razao) or abs(razao - 1.0) > 1.0e-8:
            avisar(f"limite de onda longa nao confirmado para {nome}: mstar/m={razao:.12g}.")


def main():
    criar_diretorio_saida()
    validar_limite_ondas_longas()
    metricas_freq = calcular_metricas_frequencia()
    ordens_globais, ordens_pares = estimar_ordens_convergencia(metricas_freq)
    metricas_cg = calcular_metricas_cgz()

    salvar_csv(OUTPUT_DIR / "metricas_frequencia.csv", metricas_freq,
               ["esquema", "pontos_por_comprimento_de_onda", "dz_m", "theta",
                "erro_mstar", "L2_omega", "Linf_omega", "bias_omega"])
    salvar_csv(OUTPUT_DIR / "ordem_convergencia_global.csv", ordens_globais,
               ["esquema", "ordem_global", "pontos_inicio", "pontos_fim"])
    salvar_csv(OUTPUT_DIR / "ordem_convergencia_pares.csv", ordens_pares,
               ["esquema", "pontos_inicio", "pontos_fim", "ordem_entre_pares"])
    salvar_csv(OUTPUT_DIR / "metricas_velocidade_grupo.csv", metricas_cg,
               ["esquema", "pontos_por_comprimento_de_onda", "dz_m", "L2_cgz", "Linf_cgz", "bias_cgz"])

    plotar_erro_frequencia()
    plotar_convergencia_frequencia(metricas_freq, ordens_globais)
    plotar_velocidade_grupo()
    plotar_convergencia_cgz(metricas_cg)
    escrever_resumo(metricas_freq, ordens_globais, metricas_cg)

    imprimir_tabela_frequencia(metricas_freq)
    imprimir_resumo_ordens_cgz(ordens_globais, metricas_cg)
    print(f"\nResultados gravados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
