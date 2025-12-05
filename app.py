# ==============================
# Vedê Analytics - App completo
# ==============================

import streamlit as st
from streamlit_option_menu import option_menu
from pathlib import Path
import base64

# =========================
# Util: imagem base64 (logo)
# =========================
def img_to_base64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

LOGO_B64 = img_to_base64("Images/Vede_Preditiva.png")  # ajuste o caminho se necessário

# =========================
# SIDEBAR (estilizada)
# =========================
with st.sidebar:
    st.markdown(f"""
    <style>
        /* ====== LOGO NEON ====== */
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

        /* ====== NOME ====== */
        .restaurant-name {{
            text-align: center;
            font-size: 22px;
            font-weight: 600;
            letter-spacing: 2px;
            color: #fafd07;
            margin-bottom: 20px;
            margin-top: 5px;
        }}

        /* ====== AJUSTE MENU ====== */
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
        .nav-link i {{ filter: drop-shadow(0 0 2px black); }}

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

    <!-- LOGO -->
    <img class="sidebar-logo" src="data:image/png;base64,{LOGO_B64}">
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

# =========================
# PÁGINAS COMO FUNÇÕES
# =========================
def page_inicio():
    st.markdown(
        """
        <h1 style='text-align: center; color: #fffafa;'>🍻 Bem-vindo ao Vedê Analytics!</h1>
        """, unsafe_allow_html=True
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
                    font-size: 18px; margin-top: 20px; color:#fff;'>
            🔍 <strong>Analytics do Vedê Bar</strong><br>
            Acompanhe indicadores e consuma insights para melhorar a gestão do bar.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.write("")
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; font-size: 16px; margin-top: 15px; color: #bbb;'>Utilize o menu ao lado para navegar!</div>",
        unsafe_allow_html=True
    )

def page_dashboard():
    st.title("Dashboard de Resultados")
    # Link seguro (Embed para a organização)
    PBI_ORG_URL = "https://app.powerbi.com/reportEmbed?reportId=4228da85-3150-4446-b859-bc6a523ff392&autoAuth=true&ctid=99e4b1c3-59a8-4500-8202-462ba4069af1"

    st.components.v1.html(
        f"""
        <div style="position:relative; width:100%; max-width: 800px; margin: 0 auto;">
            <div style="position:relative; padding-top:56.25%; border-radius:14px; overflow:hidden; box-shadow:0 10px 24px rgba(0,0,0,.12);">
                <iframe
                    title="Vedê Bar - Dashboard Premium"
                    src="{PBI_ORG_URL}&filterPaneEnabled=false&navContentPaneEnabled=true"
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

def page_pred():
    # =========================
    # Imports locais da aba
    # =========================
    import os, hashlib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    st.title("Análise Preditiva")

    # ========= helpers de cache (NÃO mudam o modelo) =========
    def _file_signature(path: str) -> str:
        stat = os.stat(path)
        base = f"{path}|{stat.st_size}|{stat.st_mtime}"
        return hashlib.md5(base.encode()).hexdigest()

    @st.cache_data(show_spinner=False)
    def load_csv_cached(path: str, delimiter: str) -> pd.DataFrame:
        return pd.read_csv(path, delimiter=delimiter)

    MONETARY_COLS = ['Valor_Unitaro', 'Valor_de_Desconto', 'Valor_total']

    def calculate_mape(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        mask = y_true > 0
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @st.cache_data(show_spinner=False)
    def clean_monetary_cols(df):
        df = df.copy()
        for col in MONETARY_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @st.cache_data(show_spinner=False)
    def prepare_data_full(df):
        df = clean_monetary_cols(df)

        df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', errors='coerce')
        df['Hora'] = df.get('Hora', '').astype(str)

        df['Data_Hora_Pedido'] = pd.to_datetime(
            df['Data'].dt.strftime('%d/%m/%Y') + ' ' + df['Hora'],
            format='%d/%m/%Y %H:%M:%S', errors='coerce'
        )

        df['Dia_Semana'] = df['Data_Hora_Pedido'].dt.day_name()
        df['Mes'] = df['Data_Hora_Pedido'].dt.month
        df['Hora_do_Dia'] = df['Data_Hora_Pedido'].dt.hour
        df['Dia_Util'] = df['Data_Hora_Pedido'].dt.weekday < 5
        df['Duracao_min'] = 0
        df['Tem_Desconto'] = np.where(df.get('Valor_de_Desconto', 0) > 0, 1, 0)

        if {'Cliente', 'id_pedido'}.issubset(df.columns):
            df['Frequencia_Cliente'] = df.groupby('Cliente')['id_pedido'].transform('count')
        else:
            df['Frequencia_Cliente'] = 0

        colunas_a_remover = [
            'id_pedido','Documento','Data_Evento','Hora_Evento','Data_Hora_Pedido',
            'Cliente','Vendedor','Tipo_da_Transacao'
        ]
        df_tratado = df.drop(colunas_a_remover, axis=1, errors='ignore')
        return df_tratado.dropna(subset=['Data'])

    @st.cache_data(show_spinner=False)
    def prepare_for_prediction(df_sim, _cols_train):
        cols_train = list(_cols_train)  # <<< solução oficial p/ cache
        df_sim = clean_monetary_cols(df_sim.copy())

        df_sim['Data'] = pd.to_datetime(df_sim['Data'], errors='coerce')
        df_sim['Hora'] = df_sim.get('Hora', '').astype(str)
        df_sim['Data_Hora_Pedido'] = pd.to_datetime(
            df_sim['Data'].dt.strftime('%d/%m/%Y') + ' ' + df_sim['Hora'],
            format='%d/%m/%Y %H:%M:%S', errors='coerce'
        )
        df_sim['Dia_Semana'] = df_sim['Data_Hora_Pedido'].dt.day_name()
        df_sim['Mes'] = df_sim['Data_Hora_Pedido'].dt.month
        df_sim['Hora_do_Dia'] = df_sim['Data_Hora_Pedido'].dt.hour
        df_sim['Dia_Util'] = df_sim['Data_Hora_Pedido'].dt.weekday < 5
        df_sim['Duracao_min'] = 0
        df_sim['Tem_Desconto'] = np.where(df_sim.get('Valor_de_Desconto', 0) > 0, 1, 0)

        if {'Cliente', 'id_pedido'}.issubset(df_sim.columns):
            df_sim['Frequencia_Cliente'] = df_sim.groupby('Cliente')['id_pedido'].transform('count')
        else:
            df_sim['Frequencia_Cliente'] = 0

        df_final = df_sim.drop(columns=[
            'Valor_total','Data','Hora','id_pedido','Documento',
            'Data_Evento','Hora_Evento','Data_Hora_Pedido','Cliente',
            'Vendedor','Tipo_da_Transacao'
        ], errors='ignore')

        cols_intersection = [c for c in cols_train if c in df_final.columns]
        return df_final[cols_intersection]

    # ======== Ler CSV (cache) ========
    CSV_PATH = 'pedidos_base_tratada.csv'
    import os as _os
    if not _os.path.exists(CSV_PATH):
        st.error(f"Arquivo não encontrado: {CSV_PATH}")
        st.stop()

    df_original = load_csv_cached(CSV_PATH, delimiter=';')
    _ = _file_signature(CSV_PATH)

    # ===================== MODELO (inalterado) =====================
    df_tratado = prepare_data_full(df_original.copy())

    TARGET = 'Valor_total'
    X = df_tratado.drop(TARGET, axis=1).drop(columns=['Data', 'Hora'], errors='ignore')
    y = df_tratado[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    colunas_categoricas = X.select_dtypes(include=['object', 'bool']).columns
    colunas_numericas = X.select_dtypes(include=['float64', 'int64']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), colunas_categoricas),
            ('num', StandardScaler(), colunas_numericas)
        ],
        remainder='passthrough'
    )

    best_params = {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 2}

    final_rf_model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=best_params['n_estimators'],
            max_depth=best_params['max_depth'],
            min_samples_split=best_params['min_samples_split'],
            random_state=42,
            n_jobs=-1
        ))
    ])

    final_rf_model.fit(X_train, y_train)

    # Métricas
    final_predictions = final_rf_model.predict(X_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = float(np.sqrt(final_mse))
    final_r2 = float(r2_score(y_test, final_predictions))
    final_mape = float(calculate_mape(y_test.values, final_predictions))

    st.subheader("Métricas do Modelo")
    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{final_r2:.4f}")
    c2.metric("RMSE (R$)", f"{final_rmse:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    c3.metric("MAPE", f"{final_mape:.2f}%")

    # ===================== PROJEÇÃO (inalterada) =====================
    df_limpo = prepare_data_full(df_original.copy())
    df_limpo['Mes_Ano'] = df_limpo['Data'].dt.to_period('M')
    receita_mensal_historica = df_limpo.groupby('Mes_Ano')['Valor_total'].sum().reset_index()
    receita_mensal_historica['Mes_Ano'] = receita_mensal_historica['Mes_Ano'].astype(str)

    df_outubro_completo = df_limpo[df_limpo['Mes_Ano'] == '2025-10'].copy()

    if not df_outubro_completo.empty:
        ultimo_dia_real = df_outubro_completo['Data'].max()
        dias_faltantes = pd.to_datetime(pd.date_range(start=ultimo_dia_real + pd.Timedelta(days=1), end='2025-10-31'))
        n_dias_faltantes = len(dias_faltantes)
        media_pedidos_por_dia = df_limpo.groupby('Data').size().mean()
        n_transacoes_simuladas = int(media_pedidos_por_dia * n_dias_faltantes)

        df_simulacao_out = df_original.sample(n=n_transacoes_simuladas, replace=True, random_state=42).reset_index(drop=True)
        df_simulacao_out['Data'] = np.random.choice(dias_faltantes, size=n_transacoes_simuladas)

        X_simulado_out = prepare_for_prediction(df_simulacao_out.copy(), X.columns)
        previsoes_out_imputadas = final_rf_model.predict(X_simulado_out) 
        df_simulacao_out['Valor_total'] = previsoes_out_imputadas

        receita_imputada_out = df_simulacao_out.groupby(df_simulacao_out['Data'].dt.to_period('M'))['Valor_total'].sum().reset_index()
        receita_out_real = df_outubro_completo.groupby(df_outubro_completo['Data'].dt.to_period('M'))['Valor_total'].sum().reset_index()
        
        total_outubro = receita_out_real['Valor_total'].sum() + receita_imputada_out['Valor_total'].sum()
        
        receita_mensal_historica_base = receita_mensal_historica[receita_mensal_historica['Mes_Ano'] < '2025-10']
        receita_outubro_consolidada = pd.DataFrame({'Mes_Ano': ['2025-10'], 'Valor_total': [total_outubro]})
        
    else:
        total_outubro = receita_mensal_historica['Valor_total'].iloc[-1]
        receita_mensal_historica_base = receita_mensal_historica
        receita_outubro_consolidada = pd.DataFrame()

    crescimento_nov = 1.05  
    crescimento_dez = 1.15  
    queda_jan = 0.95      

    receita_prevista_nov = total_outubro * crescimento_nov
    receita_prevista_dez = receita_prevista_nov * crescimento_dez
    receita_prevista_jan = receita_prevista_dez * queda_jan

    receita_futura_projecao = pd.DataFrame({
        'Mes_Ano': ['2025-11', '2025-12', '2026-01'],
        'Valor_total': [receita_prevista_nov, receita_prevista_dez, receita_prevista_jan]
    })

    df_final = pd.concat([
        receita_mensal_historica_base,
        receita_outubro_consolidada,
        receita_futura_projecao
    ], ignore_index=True)

    df_final['Tipo'] = (
        ['Real'] * len(receita_mensal_historica_base) +
        ['Imputado/Real'] * len(receita_outubro_consolidada) +
        ['Projeção Estratégica'] * len(receita_futura_projecao)
    )

    # ===================== VISUALIZAÇÃO (transparente + branco) =====================
    st.subheader("Receita Total Mensal: Histórico e Projeção")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Fundo transparente
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    fig.subplots_adjust(top=0.9)

    # Linha principal branca
    sns.lineplot(
        x='Mes_Ano',
        y='Valor_total',
        data=df_final,
        marker='o',
        markersize=8,
        linewidth=3,
        color='white',
        alpha=0.85,
        ax=ax
    )

    cores_destaque = {
        'Real': '#1E90FF',          # azul vibrante
        'Imputado/Real': '#FFA500', # laranja
        'Projeção Estratégica': '#00FF7F'  # verde
    }

    for tipo, cor in cores_destaque.items():
        subset = df_final[df_final['Tipo'] == tipo]
        if not subset.empty:
            sns.scatterplot(
                x='Mes_Ano',
                y='Valor_total',
                data=subset,
                color=cor,
                s=150,
                label=tipo,
                zorder=5,
                ax=ax 
            )

    ymax = df_final['Valor_total'].max() if not df_final.empty else 0
    for _, row in df_final.iterrows():
        valor_formatado = f"{row['Valor_total']:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        ax.text(
            row['Mes_Ano'], 
            row['Valor_total'] + ymax * 0.01,
            f"R$ {valor_formatado}", 
            ha='center', va='bottom', fontsize=10, color='white', fontweight='bold'
        )

    indice_divisao = len(receita_mensal_historica_base) + len(receita_outubro_consolidada) - 0.5
    if indice_divisao >= 0:
        ax.axvline(
            x=indice_divisao,
            color='white',
            linestyle='--',
            linewidth=1.5,
            label='Início da Projeção'
        )

    # Tudo branco nos eixos/título/legenda
    ax.set_title('Receita Total Mensal: Histórico e Projeção', fontsize=16, color='white')
    ax.set_xlabel('Mês/Ano', fontsize=12, color='white')
    ax.set_ylabel('Receita Total (R$)', fontsize=12, color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    leg = ax.legend(title='Tipo de Dado')
    if leg:
        plt.setp(leg.get_texts(), color='white')
        plt.setp(leg.get_title(), color='white')
        leg.get_frame().set_edgecolor('white')
        leg.get_frame().set_facecolor('none')

    ax.grid(True, linestyle='--', alpha=0.3, color='white')
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)

# =========================
# ROTEADOR
# =========================
if selected == "Início":
    page_inicio()
elif selected == "Dashboard":
    page_dashboard()
elif selected == "Análise Preditiva":
    page_pred()
