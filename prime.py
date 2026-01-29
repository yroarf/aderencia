import streamlit as st
import trafilatura
from urllib.parse import urljoin, urlparse
from groq import Groq
import os
import pandas as pd
from lxml import html
from trafilatura import html2txt
from bs4 import BeautifulSoup
import json
import matplotlib.pyplot as plt
import re
import tempfile
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat

from rascunho import extrair_subdominio_gov

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========================= CONFIGURAÇÃO PÁGINA =========================

st.set_page_config(
    page_title=" Analisador de Aderência",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    '/login', '/cadastro', '/conta', '/privacidade',
    '/contato', '/sobre', '/equipe', '/assinatura',
    '/webmail', '/galeria', '/simbolos'
          ]

# ======================= PROMPT DE ANÁLISE =============

prompt_padrao = f"""
Você é um analista jurídico especializado em direito eleitoral.

Analise o texto fornecido e identifique qualquer indício de conduta vedada aos agentes públicos **em período eleitoral**, considerando a data de referência: {data_referencia}.

As vedações e restrições devem ser avaliadas com base no momento próximo à data informada ({data_referencia}).

Considere na análise a base legal ({base_legal}) de referência informada pelo usuário.

REGRAS OBRIGATÓRIAS:
- Não adicione explicações, introduções ou conclusões.
- Não use markdown, negrito ou qualquer formatação.
- Retorne APENAN um JSON válido com a estrutura exata abaixo.

FORMATO OBRIGATÓRIO DO RETORNO (nada mais, nada menos):

[
  {{"item": 1, "trecho": "texto exato do site", "classificacao": "divergente"}},
  {{"item": 2, "trecho": "outro trecho", "classificacao": "indicio"}}
]

[28, 25, 2, 1]

Se não houver divergências ou indícios, retorne:

[]

[total_trechos, total_trechos, 0, 0]

Texto para análise:
\"\"\"{chunk}\"\"\"
"""

col_titulo, col_data = st.columns(2)
with col_titulo:
    st.title("🗳️ Analisador de Aderência")
with col_data:
    st.markdown("**Data de referência**")
    data_referencia = st.date_input(
        label="Período eleitoral de referência",
        value=None,  # sem valor padrão fixo → usuário deve escolher
        min_value=None,
        max_value=None,
        help="Selecione a data do primeiro turno).",
        format="DD/MM/YYYY"
    )
if data_referencia is not None:
    st.session_state.data_referencia = data_referencia
    st.caption(f"Data selecionada: **{data_referencia.strftime('%d/%m/%Y')}**")
else:
    st.session_state.data_referencia = None
    st.info("Selecione uma data de referência para ativar a análise contextualizada no período eleitoral.")

st.markdown("**Compare conteúdo de notícias de sites institucionais com normas eleitorais**")
st.divider()

# Divisor visual
st.divider()

# ========================================================================================
#                                Adição e Lista de Sites (ok)
# ========================================================================================

st.markdown("### Adição de Sites")

if "sites_df" not in st.session_state:
    st.session_state.sites_df = pd.DataFrame(columns=["URL", "Nome do Site"])

# ====================== ADIÇÃO DE NOVO SITE ======================
def extrair_subdominio_gov(url: str) -> str:

    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()

    if ':' in netloc:
        netloc = netloc.split(':')[0]
    if netloc.startswith('www.'):
        netloc = netloc[4:]
    if not netloc.endswith('.gov.br'):
        raise ValueError(f"A URL não termina com .gov.br: {url}")
    dominio_sem_gov = netloc[:-7]
    partes = dominio_sem_gov.split('.')
    if len(partes) >= 2:
        resultado = '.'.join(partes[-2:])
    else:
        resultado = partes[-1]
    return resultado


with st.expander("🌐 sites", expanded=False):
    st.markdown("##### Adicionar novo site")
    col1, col2 = st.columns([3, 1])
    with col1:
        nova_url = st.text_input(
            "URL do site (ex: https://www.municipio.uf.gov.br/noticias)",
            placeholder="https://www.exemplo.go.gov.br/noticias",
            help="Página principal de notícias ou comunicados do município."
        )

    if st.button("Adicionar Site", type="primary"):
        if not nova_url.strip():
            st.error("Por favor, insira uma URL válida.")
        else:
            url_limpa = nova_url.strip().rstrip("/")
            urls_existentes = st.session_state.sites_df["URL"].str.rstrip("/").tolist()

            if url_limpa in urls_existentes:
                st.error("Esta URL já foi adicionada.")
            else:
                nome_exibicao = urlparse(url_limpa).netloc
                novo_site = pd.DataFrame([{
                    "URL": url_limpa,
                    "Nome do Site": nome_exibicao
                }])
                st.session_state.sites_df = pd.concat(
                    [st.session_state.sites_df, novo_site],
                    ignore_index=True
                )
                st.success(f"Site adicionado: {nome_exibicao}")
                st.rerun()

# ====================== LISTA EDITÁVEL DE SITES ======================
    st.markdown("##### Lista de Sites para Análise")

    if st.session_state.sites_df.empty:
        st.info("Nenhum site adicionado ainda. Use o campo acima para incluir.")
    else:
        # Data editor com validação de duplicatas
        edited_df = st.data_editor(
            st.session_state.sites_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "URL": st.column_config.TextColumn(
                    "URL",
                    required=True,
                    help="URL completa da página de notícias"
                ),
                "Nome do Site": st.column_config.TextColumn(
                    "Nome do Site",
                    required=True,
                    help="Nome amigável para exibição"
                )
            },
            hide_index=True
        )

        # Validação: impedir URLs duplicadas ao editar
        urls_editadas = edited_df["URL"].str.strip().str.rstrip("/").tolist()
        if len(urls_editadas) != len(set(urls_editadas)):
            st.error("⚠️ Atenção: Não é permitido ter URLs duplicadas na lista.")
        else:
            # Só atualiza o estado se não houver duplicatas
            st.session_state.sites_df = edited_df
            st.success("Lista atualizada com sucesso!")
        # print(edited_df)
        st.caption(f"Total de sites: **{len(st.session_state.sites_df)}**")


st.markdown("### **Base Legal**")
with st.expander("📋 Base Legal", expanded=False):
    st.markdown("Defina o texto de referência legal que será usado na análise de aderência pelo LLM.")

    # Abas para separar as funcionalidades
    tab_txt, tab_pdf_converter = st.tabs(
        [" Referência em TXT (até 5 arquivos)", " Conversor PDF → TXT (ferramenta isolada)"])

    # ===============================================
    # ABA 1: Carregar múltiplos TXT como referência
    # ===============================================
    with tab_txt:
        st.markdown("### Upload arquivos .txt")
        st.markdown("**Carregue até 5 arquivos .txt** com trechos da lei, resolução, portaria, cartilha etc.")

        uploaded_txt_files = st.file_uploader(
            "Selecione arquivos TXT",
            type=["txt"],
            accept_multiple_files=True,
            key="txt_referencia_multi",
            help="Máximo de 5 arquivos. Todos serão combinados em um único texto para a análise."
        )

        texto_referencia = ""

        if uploaded_txt_files:
            if len(uploaded_txt_files) > 5:
                st.error("Limite máximo: 5 arquivos TXT.")
                uploaded_txt_files = uploaded_txt_files[:5]

            textos_carregados = []
            for file in uploaded_txt_files:
                try:
                    content = file.read().decode("utf-8")
                    textos_carregados.append(f"\n\n=== Conteúdo de: {file.name} ===\n{content}")
                except Exception as e:
                    st.warning(f"Erro ao ler {file.name}: {e}")

            if textos_carregados:
                texto_referencia = "\n".join(textos_carregados)
                st.success(f"{len(textos_carregados)} arquivo(s) TXT carregado(s) com sucesso.")
                st.caption(f"Total de caracteres: {len(texto_referencia):,}")

        # Campo opcional para texto manual
        st.markdown("**Ou cole texto diretamente (opcional)**")
        texto_manual = st.text_area(
            "Texto adicional ou complementar",
            height=150,
            placeholder="Cole aqui trechos específicos, artigos isolados etc."
        )

        # Texto final consolidado para a LLM
        referencia_final = texto_referencia
        if texto_manual.strip():
            referencia_final += "\n\n" + texto_manual.strip() #adiciona o texto inserido no text_area

        if not referencia_final.strip():
            st.warning("Nenhum texto de referência carregado ainda.")
        else:
            st.info("Texto de referência pronto para uso na análise.")

    # ===============================================
    # ABA 2: Conversor isolado PDF → TXT
    # ===============================================
    with tab_pdf_converter:
        st.markdown("### 🔄 Ferramenta Isolada: Converter PDF para TXT")
        st.info(
            "Esta ferramenta **não afeta** a base legal principal. Ela apenas converte um PDF em TXT para download.")

        pdf_file = st.file_uploader(
            "Faça upload do PDF para conversão",
            type=["pdf"],
            key="pdf_converter_isolado"
        )

        nome_arquivo_saida = st.text_input(
            "Nome do arquivo TXT de saída (sem extensão)",
            value="texto_extraido_pdf",
            help="O arquivo será salvo como 'nome.txt'"
        )

        if pdf_file and st.button("Converter PDF para TXT", type="secondary"):
            with st.spinner("Convertendo PDF para texto..."):
                try:
                    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=True)
                    converter = DocumentConverter(
                        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
                    )

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        tmp_path = tmp_file.name

                    result = converter.convert(tmp_path)
                    os.unlink(tmp_path)

                    textos = [t["text"] for t in result.document.export_to_dict().get("texts", [])]
                    texto_convertido = "\n".join(textos)

                    st.success("Conversão concluída!")

                    # Download do TXT
                    nome_final = f"{nome_arquivo_saida.strip()}.txt"
                    st.download_button(
                        label="📥 Baixar arquivo TXT",
                        data=texto_convertido.encode("utf-8"),
                        file_name=nome_final,
                        mime="text/plain"
                    )

                    st.caption(f"Caracteres extraídos: {len(texto_convertido):,}")

                except Exception as e:
                    st.error(f"Erro na conversão: {str(e)}")

st.markdown("### **Prompt**" )
with st.expander("🧠 Prompt", expanded=False):

    st.markdown("#### Prompt para Análise")
    prompt_personalizado = st.text_area(
        "Edite o prompt que será enviado ao modelo",
        value=prompt_padrao,
        height=350,
        key="prompt_editor"
    )

    # Variável global para uso na análise

base_legal = referencia_final if 'referencia_final' in locals() else ""


st.divider()

# ========================================================================================
#                            Configurações do Modelo de IA
# ========================================================================================
st.markdown("### IA (LLM)")
with st.expander("🤖 **Configurações do Modelo de IA**", expanded=False):
    col_model1, col_model2 = st.columns(2)
    with col_model1:
        modeloIA = st.selectbox(
            "Selecione o Modelo de IA (Groq)",
            options=GROQ_MODELS,
            index=0
        )

    with col_model2:
        max_links = st.slider("Número máximo de LINKS por URL", 1, 20, 5, help="Quantos links internos seguir por site")

    col_chunk, col_temperatura = st.columns(2)
    with col_chunk:
        max_chunks = st.slider("Número máximo de CHUNKS por URL", 1, 2000, 100,
                               help="Chunks maiores = análise mais profunda, mas mais lenta")
    with col_temperatura:
        temperatura = st.slider("Temperatura (criatividade)", 0.0, 2.0, 0.7, 0.1, help="O valor 0.0 é determinístico")

st.divider()

# ============================================================
#  COLETA DE LINKS DO SITE (ok)
# ============================================================

def coletar_links_internos(url: str, max_links) -> set:
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
        # print(max_links)
        if len(links_validos) >= max_links:
            break
    # print(links_validos)
    return links_validos

# ============================================================
#             EXTRAÇÃO DE TEXTO     (ok)
# ============================================================

def extrair_texto(url: str) -> str:
    # 1. BAIXA HTML BRUTO — ESSENCIAL (ok)

    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        print(f"[ERRO] Falha ao baixar HTML bruto: {url}")
        return ""

    texto_final = None

    # 2. PRIMEIRA TENTATIVA — Trafilatura com máximo recall

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

    # 3. FALLBACK 1 — html2txt (Trafilatura modo bruto)

    if not texto_final:
        try:
            print(f"[FALLBACK] html2txt ativado para {url}")
            raw_text = html2txt(downloaded)
            if raw_text and len(raw_text.strip()) > 100:
                texto_final = raw_text
        except:
            pass

    # 4. FALLBACK 2 — BeautifulSoup (captura TODO texto visível)

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

    # print("texto extração")
    # print(texto_final)

    return texto_final


# ============================================================
#  CHUNKING POR PARÁGRAFOS (ok)
# ============================================================

def chunk_por_paragrafos(texto, limite):
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
#  ANÁLISE COM LLM - chamada da API do Groq (ok)
# ============================================================

def analisar_com_llm(chunk: str, model: str, temperatura: float, prompt_base: str, referencia_legal: str, data_referencia: str):

    if not chunk.strip():
        return [], [0, 0, 0, 0]
    prompt_completo = prompt_base.format(
        chunk=chunk,
        referencia_legal=base_legal or "Nenhuma base legal fornecida.",
        data_referencia=data_referencia,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_completo}],
            temperature=temperatura,
            max_completion_tokens=1024
        )
        content = response.choices[0].message.content.strip()
        # print(content)

        # === Extração da lista de contagem ===
        contagem = [0, 0, 0, 0]
        match = re.search(r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', content)
        if match:
            contagem = [int(match.group(i)) for i in range(1, 5)]
        trechos = []
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))

                if isinstance(data, list):
                    for item in data:
                        trecho = item.get("trecho") or item.get("texto") or item.get("Trecho")
                        classificacao = item.get("classificacao") or item.get("classificação") or item.get("tipo")

                        if trecho:
                            trechos.append({
                                "trecho": trecho.strip(),
                                "classificacao": classificacao or "indefinido"
                            })

            except Exception as e:
                print("JSON inválido:", e)

        return trechos, contagem

    except Exception as e:
        st.warning(f"Erro na chamada ao LLM: {e}")
        return [], [0, 0, 0, 0]


# ========================================================================================
#                                    ANÁLISE DOS SITES
# ========================================================================================


if "resultados" not in st.session_state:
    st.session_state.resultados = []

colAnalisar1, colAnalisar2, colAnalisar3 = st.columns([1, 2, 1])
with colAnalisar2:
    analisar = st.button("🚀 **Analisar Sites**", type="primary", use_container_width=True)

if analisar:
    if st.session_state.sites_df.empty:
        st.error("Adicione pelo menos um site antes de analisar.")
    else:
        sites = st.session_state.sites_df.to_dict("records")  # ok
        # print(sites)
        resultados = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        # print(sites)
        for idx, site in enumerate(sites):
            url = site["URL"]
            status_text.text(f"Analisando {idx + 1}/{len(sites)}: {url}")
            # print(max_links)
            links = coletar_links_internos(url, max_links=max_links)

            total_trechos_global = 0
            aderentes_global = 0
            divergentes_global = 0
            indicios_global = 0
            # textos_completos = []
            trechos_divergentes = []
            # print(links)
            for link in links:
                texto = extrair_texto(link)
                if texto and len(texto.split()) > 10:  # Busca detalhada até o nível de frases.
                    chunks = chunk_por_paragrafos(texto, limite=max_chunks)
                    # print(chunks)
                    for chunk in chunks:
                        if chunk.strip():
                            trecho_divergente, lista_contagem = analisar_com_llm(
                                chunk,
                                modeloIA,
                                temperatura,
                                prompt_personalizado,
                                base_legal,
                                data_referencia=st.session_state.get("data_referencia"))

                            if trecho_divergente:
                                trechos_divergentes.extend(trecho_divergente)

                            total_trechos_global += lista_contagem[0]
                            aderentes_global += lista_contagem[1]
                            divergentes_global += lista_contagem[2]
                            indicios_global += lista_contagem[3]

            # Calcula percentual de aderência da URL
            if total_trechos_global == 0:
                percAderencia = 0.0
            else:
                percAderencia = round((aderentes_global / total_trechos_global) * 100, 1)

            resultados.append({

                "url": url,
                "aderencia": percAderencia,
                "total_trechos": total_trechos_global,
                "aderentes": aderentes_global,
                "trechos divergentes": [trechos_divergentes]

            })
            print(resultados)

            progress_bar.progress((idx + 1) / len(sites))

        status_text.empty()
        progress_bar.empty()
        st.session_state.resultados = resultados

# =====================================================================
# ========================= GRÁFICO DE BARRAS =========================
# =====================================================================



resultados_para_plot = st.session_state.get("resultados", [])

if resultados_para_plot:
    def nome_grafico(url):
        return extrair_subdominio_gov(url)

    df_result = pd.DataFrame({
        "Site": [nome_grafico(r.get("url", "")) for r in resultados_para_plot],
        "Aderencia (%)": [float(r.get("aderencia", 0.0)) for r in resultados_para_plot]
    })
    print('df_result')
    print(df_result)
    # Remove entradas vazias (defensivo)
    df_result = df_result.dropna(subset=["Site", "Aderencia (%)"])

    if not df_result.empty:
        col_esq, col_centro, col_dir = st.columns([1, 2, 1])

        with col_centro:
            fig, ax = plt.subplots(figsize=(10, 5))

            sites = df_result["Site"]
            valores = df_result["Aderencia (%)"].astype(float).clip(0, 100)

            # Cores por gradiente
            cores = plt.cm.viridis(valores / 100.0)

            bars = ax.bar(sites, valores, color=cores, edgecolor='blue', linewidth=0.8)

            # Rótulos com percentual
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f'{height:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

            ax.set_xlabel("")
            ax.set_ylabel("Aderência (%)", fontsize=10)
            ax.set_title(" 📊 Grau Aderência", fontsize=10, pad=20)

            ax.tick_params(axis='x', labelsize=8, rotation=45)
            ax.tick_params(axis='y', labelsize=8)

            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylim(0, 100)

            plt.tight_layout()
            st.pyplot(fig)

# Rodapé
st.markdown("---")
st.caption("Analisador de Aderência | Desenvolvido por Fabiana, João Vicente, Lívia, Túlio e Yroá")
