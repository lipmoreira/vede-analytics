import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import base64

# --- util: carregar imagem local como base64 ---
def img_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

LOGO_B64 = img_to_base64("Images/Vede_Preditiva.png")  # caminho local

with st.sidebar:
    st.markdown(f"""
    <style>


        /* ================= LOGO NEON ================= */
        .sidebar-logo {{
            width: 130px;
            margin: 25px auto 10px auto;
            display: block;
            filter: drop-shadow(0 0 6px #FFD700)
                    drop-shadow(0 0 16px rgba(255,215,0,0.8));
            animation: pulseNeon 2.5s infinite alternate;
        }}

        @keyframes pulseNeon {{
            from {{ filter: drop-shadow(0 0 6px #FFD700); }}
            to   {{ filter: drop-shadow(0 0 14px #FFD700) drop-shadow(0 0 28px rgba(255,215,0,0.8)); }}
        }}

        /* ====== NOME DO BAR ====== */
        .restaurant-name {{
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            letter-spacing: 2px;
            color: #fafd07;
            margin-bottom: 20px;
            margin-top: 5px;
        }}

        /* ====== AJUSTES NO OPTION_MENU ====== */
        .nav-link {{
            font-size: 15px !important;
            padding: 14px !important;
            border-radius: 12px !important;
            margin-bottom: 8px !important;
            color: #e0e0e0 !important;
            transition: 0.3s ease;
            background: rgba(255,255,255,0.03) !important;
        }}

        .nav-link:hover {{
            background: linear-gradient(90deg, #FFD70033, #00000000) !important;
            transform: translateX(5px);
            color: #FFD700 !important;
        }}

        .nav-link.active {{
            background: linear-gradient(90deg, #ff8c00aa, #00000000) !important;
            color: #FFA500 !important;
            font-weight: 600 !important;
            box-shadow: 0 0 10px rgba(255,140,0,0.3);
        }}


        .nav-link i {{
            filter: drop-shadow(0 0 2px black);
        }}

        /* ====== RODAPÉ ====== */
        .sidebar-footer {{
            text-align: center;
            font-size: 12px;
            color: #888;
            margin-top: 40px;
            padding-top: 12px;
            border-top: 1px solid #333;
        }}
    </style>

    <!-- LOGO (via base64) -->
    <img class="sidebar-logo" src="data:image/png;base64,{LOGO_B64}">

    <!-- NOME -->
    <div class="restaurant-name">vedê bar</div>
    """, unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Início", "Dashboard", "Análise Preditiva"],
        icons=["house-fill", "window-stack", "speedometer"],
        default_index=0,
    )

    st.markdown("""
    <div class="sidebar-footer">
        Sistema Analytics<br>
        © 2025 Vedê Bar
    </div>
    """, unsafe_allow_html=True)

if selected == "Início":
    st.markdown(
        """
        <h1 style='text-align: center; color: #fffafa;'>
            🍻 Bem-vindo ao Vedê Analytics!
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style='display: flex; justify-content: center; margin-top: 20px;'>
            <img src='https://lh3.googleusercontent.com/gps-cs-s/AG0ilSxfHIGbwAykWwr_e-U7Ru5o2qS-yLJBK5dIfGT8_Qyys-ECJW39yzGdVBIE93-DKT-Y8rjOlnT1SL9F8Z61geM4Jo_RdHkzxSKy7IXBefgmWVoXpLdUxuPhE8yQG1raHUmzHt83=s1360-w1360-h1020-rw'
                 style='width: 80%; border-radius: 18px;'>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")
    st.markdown(
        """
        <div style='padding: 18px; border-radius: 15px;
                    background-color: rgba(224,164,0,0.15);
                    border-left: 8px solid #E0A400;
                    font-size: 18px; margin-top: 20px;'>
            🔍 <strong>Analytics do Vedê Bar</strong><br>
            Acompanhe indicadores e consuma insights para melhorar a gestão do bar.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 16px; margin-top: 15px; color: #555;'>
            Utilize o menu ao lado para navegar!
        </div>
        """,
        unsafe_allow_html=True
    )

if selected == "Dashboard":
    st.title(f"{selected} de Resultados")

    # Link seguro (Embed para a organização)
    PBI_ORG_URL = "https://app.powerbi.com/reportEmbed?reportId=4228da85-3150-4446-b859-bc6a523ff392&autoAuth=true&ctid=99e4b1c3-59a8-4500-8202-462ba4069af1"

    st.components.v1.html(
        f"""
        <div style="position:relative; width:100%; max-width: 800px; margin: 0 auto;">
            <div style="position:relative; padding-top:56.25%; border-radius:14px; overflow:hidden; box-shadow:0 10px 24px rgba(0,0,0,.12);">
                <iframe
                    title="Vedê Bar - Dashboard Premium"
                    src="{PBI_ORG_URL}&filterPaneEnabled=false&navContentPaneEnabled=false"
                    frameborder="0"
                    allowFullScreen="true"
                    style="position:absolute; top:0; left:0; width:100%; height:100%; border:0;">
                </iframe>
            </div>
            <p style="text-align:center; color:#666; font-size:0.9rem; margin-top:10px;">
                Dica: use o celular na horizontal para ver mais detalhes.
            </p>
        </div>
        """,
        height=600,
    )


if selected == "Análise Preditiva":
    st.title(f"Bem vindo a página {selected}")
