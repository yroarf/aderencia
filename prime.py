import streamlit as st
import trafilatura
from urllib.parse import urljoin, urlparse
from groq import Groq
import os
import pandas as pd
from lxml import html
from trafilatura import html2txt
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import re

# ========================= CONFIGURAÇÃO ========================= (ok)
st.set_page_config(page_title="Analisador de Neutralidade Ideológica", layout="wide")

# ========================= VALIDAÇÃO DA CHAVE DO GROQ============ (checado)
if "GROQ_API_KEY" not in st.session_state:
    api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Chave da API do Groq não encontrada. Configure em secrets ou variável de ambiente.")
        st.stop()
    st.session_state.GROQ_API_KEY = api_key

client = Groq(api_key=st.session_state.GROQ_API_KEY)

# ========================= LISTA DE MODELOS DE IA DO GROQ========= (ok)

# alguns modelos que estão disponíves no Groq
GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "openai/gpt-oss-120b"

]

# ======================= LISTA DE CAMINHOS IRRELEVANTES ============= (ok)

LISTA_1 = [
    '/anuncie', '/publicidade', '/patrocinado', '/produto', '/loja', '/comprar',
    '/login', '/cadastro', '/conta', '/newsletter', '/termos', '/privacidade',
    '/contato', '/sobre', '/equipe', '/assinatura', '/premium', "/carros", '/futebol',
    '/esporte', '/bbb', '/carnaval', '/musica'
]

# ========================= ADIÇÃO DE SITES (TABELA EDITÁVEL) ========================= (ok)
st.title("Analisador de Neutralidade Ideológica de Sites")

st.markdown("### Adição de Sites") # ok

if "sites_df" not in st.session_state:
    st.session_state.sites_df = pd.DataFrame(columns=["URL", "Nome do Site"])

# Adicionar novo site (ok)
with st.expander("Adicionar novo site", expanded=False):
    col1, col2 = st.columns([3, 1])
    with col1:
        nova_url = st.text_input("URL do site (ex: https://exemplo.com)")
    with col2:
        novo_nome = st.text_input("Nome para exibição", value="")
    if st.button("Adicionar"):
        if nova_url:
            nome = novo_nome or urlparse(nova_url).netloc
            novo_df = pd.DataFrame([{"URL": nova_url.strip(), "Nome do Site": nome.strip()}])
            st.session_state.sites_df = pd.concat([st.session_state.sites_df, novo_df], ignore_index=True)
            st.success("Site adicionado!")
            st.rerun()

# Tabela editável (ok)
st.markdown("#### Lista de sites")
edited_df = st.data_editor(
    st.session_state.sites_df,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "URL": st.column_config.TextColumn("URL", required=True),
        "Nome do Site": st.column_config.TextColumn("Nome do Site", required=True)
    }
)

if edited_df is not None:
    st.session_state.sites_df = edited_df

# ========================= SELEÇÃO DO MODELO DE IA ============== (ok)

col1, col2 = st.columns([2, 2])

with col1:
    modelo_selecionado = st.selectbox(
        "Modelo de IA (Groq)",
        options=GROQ_MODELS,
        index=0)  # llama-3.3-70b-versatile como default
with col2:
    maxLinks = st.slider(
        "Número máximo de links por URL",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

# ============================================================
#  COLETA DE LINKS DO SITE (ok)
# ============================================================

def coletar_links_internos(url: str, max_links) -> set:
    """
    Coleta links internos utilizando HTML bruto.
    """
    # downloaded = trafilatura.fetch_url(url, output_format="raw", no_fallback=False)
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        return {url}

    try:
        tree = html.fromstring(downloaded)
    except Exception:
        return {url}

    dominio = urlparse(url).netloc
    links_validos = {url}

    for href in tree.xpath("//a/@href"):
        full = urljoin(url, href.strip())
        parsed = urlparse(full)

        if parsed.netloc != dominio:
            continue

        path = parsed.path.lower()

        if any(block in path for block in LISTA_1):
            continue

        if re.search(r'\.(pdf|jpg|jpeg|png|gif|zip|docx?|xlsx?)$', path):
            continue

        links_validos.add(full)
        print(max_links)
        if len(links_validos) >= max_links:
            break
    # print(links_validos)
    return links_validos


def limpeza_jornalistica_completa(texto: str) -> str:

    linhas = texto.splitlines()
    clean = []

    for line in linhas:
        linha = line.strip()

        if len(linha) < 10:  # Remove linhas muito curtas (menus, copyright)
            continue

        # Remove apenas propaganda explícita
        if any(palavra in linha.lower() for palavra in ["publicidade", "patrocinado", "anuncie", "assine agora", "newsletter"]):
            continue

        clean.append(linha)

    texto_final = "\n\n".join(clean)
    texto_final = re.sub(r"\n{3,}", "\n\n", texto_final).strip()

    return texto_final


# ============================================================
#  EXTRAÇÃO DE TEXTO     (ok)
# ============================================================

def extrair_texto(url: str) -> str:

    # ==========================================================
    # 1. BAIXA HTML BRUTO — ESSENCIAL (ok)
    # ==========================================================
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        print(f"[ERRO] Falha ao baixar HTML bruto: {url}")
        return ""

    texto_final = None

    # ==========================================================
    # 2. PRIMEIRA TENTATIVA — Trafilatura com máximo recall
    # ==========================================================
    try:
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_images=False,
            include_tables=True,
            deduplicate=True,
            favor_recall=True,
            favor_precision=False,
            no_fallback=False,
            include_formatting=False
        )

        if text and len(text.strip()) > 150:
            texto_final = text
        else:
            print(f"[WARN] Extração Trafilatura baixa em {url}")

    except Exception as e:
        print(f"[ERRO Trafilatura] {url}: {e}")

    # ==========================================================
    # 3. FALLBACK 1 — html2txt (Trafilatura modo bruto)
    # ==========================================================
    if not texto_final:
        try:
            print(f"[FALLBACK] html2txt ativado para {url}")
            raw_text = html2txt(downloaded)
            if raw_text and len(raw_text.strip()) > 100:
                texto_final = raw_text
        except:
            pass

    # ==========================================================
    # 4. FALLBACK 2 — BeautifulSoup (captura TODO texto visível)
    # ==========================================================
    if not texto_final:
        try:
            print(f"[FALLBACK] BeautifulSoup ativado para {url}")
            soup = BeautifulSoup(downloaded, "lxml")

            # Remove scripts, styles etc.
            for tag in soup(["script", "style", "noscript"]):
                tag.extract()

            bs_text = soup.get_text(separator="\n")
            if bs_text and len(bs_text.strip()) > 80:
                texto_final = bs_text

        except Exception as e:
            print(f"[ERRO BS4] {url}: {e}")

    if not texto_final:
        print(f"[ERRO] Nenhum método conseguiu extrair texto de {url}")
        return ""

    texto_final = limpeza_jornalistica_completa(texto_final)
    # print("texto extração")
    # print(texto_final)

    return texto_final

# ============================================================
#  CHUNKING POR PARÁGRAFOS (ok)
# ============================================================

def chunk_por_paragrafos(texto, limite=2000):
    """
    Divide texto em blocos por parágrafos, evitando cortar frases pela metade.
    """
    paragrafos = texto.split("\n")
    buffer = ""
    chunks = []

    for p in paragrafos:
        if len(buffer) + len(p) < limite:
            buffer += p + "\n"
        else:
            chunks.append(buffer)
            buffer = p + "\n"

    if buffer.strip():
        chunks.append(buffer)
    # print("chunks            _____________________")
    # print(chunks)
    return chunks

# ============================================================
#  PROMPT DO MODELO
# ============================================================



PROMPT_ANALISE = """
Você é um analista jornalístico totalmente imparcial, com formação em ciência política e jornalismo investigativo.

1. Analise o texto abaixo com base nos seguintes critérios rigorosos:

    TRECHO NEUTRO:
    - Apresenta fatos verificáveis sem adjetivação valorativa ou enquadramento persuasivo.
    - Usa linguagem descritiva, técnica ou procedimental.
    - Inclui dados, datas, nomes, números, citações diretas sem interpretação.
    - Oferece múltiplos lados com equilíbrio.
    - Emprega marcadores de incerteza adequados ("segundo...", "de acordo com...").
    
    TRECHO ENVIESADO:
    - Contém enquadramento normativo, moral ou teleológico.
    - Seleção assimétrica de fatos ou omissões relevantes.
    - Linguagem valorativa (ex: "nefasta", "salvador", "corrupto", "patriótico").
    - Atribuição causal simplificada a grupos/ideologias.
    - Rótulos, espantalhos ou generalizações.
    - Apelos emocionais predominantes.
    
    Se enviesado, classifique como:
    - ESQUERDA: ênfase em redistribuição, Estado amplo, crítica ao mercado, identitarismo, antiausteridade.
    - DIREITA: ênfase em liberdade econômica, ordem, tradição, mercado, Estado enxuto, valores familiares.

2. Elabore a resposta em uma estrutura JSON, respeitando rigorosamente o seguinte formato para cada trecho relevante:
    Para cada trecho significativo (frase ou parágrafo com ideia completa), retorne em JSON:
    
    a estrutura deve ser EXCLUSIVAMENTE com uma lista JSON válida.
    Não inclua nenhuma explicação, texto introdutório, markdown ou formatação extra.
    Exemplo de resposta correta:
    [
      {"trecho": "Texto aqui.", "classificacao": "NEUTRO", "explicacao_breve": "Fatos objetivos."}
    ]
    
3. Conte todos os trechos significativos (total_trechos) e trechos neutros (trechos_neutros).
    Responda APENAS com: total_trechos, trechos_neutros

Exemplo: 20, 18

Texto:
\"\"\"{chunk}\"\"\"
"""

# ============================================================
#  ANÁLISE COM LLM - chamada da API do Groq (ok)
# ============================================================

def analisar_com_llm(chunk: str, model: str) -> tuple[int, int]:
    """
    Retorna (total_trechos, trechos_neutros)
    """
    if not chunk.strip():
        return 0, 0

    # temperature=0.0 -> determinístico
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT_ANALISE.format(chunk=chunk)}],
            temperature=0.0,
            max_completion_tokens=512  # máximo de tokens
        )

        content = response.choices[0].message.content.strip()

        # Extrai os dois números
        # Aceita variações como "15,9" ou "15, 9"

        numeros = re.findall(r'\d+', content)
        if len(numeros) >= 2:
            total = int(numeros[0])
            neutros = int(numeros[1])
            return total, neutros
        else:
            # Fallback seguro
            return 0, 0

    except Exception as e:
        print(f"[ERRO LLM] {e}")
        return 0, 0


# ========================= xxxxxxxxxxxxxxx =========================
#                           ANÁLISE EM LOTE
# ========================= xxxxxxxxxxxxxxx =========================

if st.button("Analisar Sites", type="primary"):
    if st.session_state.sites_df.empty:
        st.warning("Adicione pelo menos um site à lista.")
        st.stop()

    sites = st.session_state.sites_df.to_dict("records") #ok
    # print(sites)
    resultados = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, site in enumerate(sites):
        url = site["URL"]
        nome = site["Nome do Site"]
        status_text.text(f"Analisando {idx + 1}/{len(sites)}: {nome}")
        # print(maxLinks)
        links = coletar_links_internos(url, max_links=maxLinks)

        total_trechos_global = 0
        neutros_global = 0
        textos_completos = []

        for link in links:
            texto = extrair_texto(link)
            if texto and len(texto.split()) > 30:  # Só analisa conteúdo relevantes, mais de 30 palavras
                textos_completos.append((link, texto))
                chunks = chunk_por_paragrafos(texto, limite=2000)

                for chunk in chunks:
                    if chunk.strip():
                        total, neutros = analisar_com_llm(chunk, modelo_selecionado)
                        total_trechos_global += total
                        neutros_global += neutros

        # Calcula neutralidade da URL
        if total_trechos_global == 0:
            neutralidade = 0.0
        else:
            neutralidade = round((neutros_global / total_trechos_global) * 100, 1)

        resultados.append({
            "nome": nome,
            "url": url,
            "neutralidade": neutralidade,
            "total_trechos": total_trechos_global,
            "neutros": neutros_global,
            "textos": textos_completos
        })

        progress_bar.progress((idx + 1) / len(sites))

    status_text.empty()
    progress_bar.empty()

    # ========================= GRÁFICO DE BARRAS =========================
    df_result = pd.DataFrame([
        {"Site": r["nome"], "Neutralidade (%)": r["neutralidade"]}
        for r in resultados[:10]
    ])

    if not df_result.empty:

        col_esq, col_centro, col_dir = st.columns([1, 2, 1])

        with col_centro:

            fig, ax = plt.subplots(figsize=(10, 5))  # Tamanho maior do gráfico

            # Dados
            sites = df_result["Site"]
            valores = df_result["Neutralidade (%)"]

            # Barras com cor gradiente
            cores = plt.cm.viridis(valores / 100)  # escala de 0 a 100%

            bars = ax.bar(sites, valores, color=cores, edgecolor='blue', linewidth=0.8)

            # Adiciona o valor %
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,  # um pouco acima da barra
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )


            ax.set_xlabel("")  # Remove label desnecessário
            ax.set_ylabel("Neutralidade (%)", fontsize=10)
            ax.set_title("Grau de Neutralidade Ideológica por Site", fontsize=10, pad=20)

            # Aumenta o tamanho dos rótulos dos sites (eixo X)
            ax.tick_params(axis='x', labelsize=8, rotation=45)  # <<< TAMANHO AUMENTADO + rotação para não sobrepor

            # Aumenta os números do eixo Y também
            ax.tick_params(axis='y', labelsize=8)

            # Grade sutil no fundo
            ax.grid(axis='y', linestyle='--', alpha=0.4)

            # Remove bordas superiores e direita para ficar mais clean
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Limite do eixo Y de 0 a 100
            ax.set_ylim(0, 100)

            # Ajusta layout para não cortar rótulos
            plt.tight_layout()

            # Exibe no Streamlit
            st.pyplot(fig)



