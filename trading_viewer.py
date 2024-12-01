import streamlit as st
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import time

def connect_db(db_name='trading_records.db'):
    """데이터베이스 연결"""
    return sqlite3.connect(db_name)

def get_available_databases():
    """사용 가능한 데이터베이스 목록 반환"""
    databases = {
        'Binance': 'trading_records.db',
        'Upbit': 'upbit_trading.db'
    }
    
    # 실제로 존재하는 데이터베이스만 필터링
    available_dbs = {}
    for name, path in databases.items():
        try:
            conn = connect_db(path)
            conn.close()
            available_dbs[name] = path
        except:
            continue
    
    return available_dbs

def get_table_columns(conn, table_name):
    """테이블의 컬럼 목록 조회"""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [col[1] for col in cursor.fetchall()]

def get_database_info(db_path):
    """데이터베이스의 테이블과 컬럼 구조 확인"""
    try:
        conn = connect_db(db_path)
        cursor = conn.cursor()
        
        # 모든 테이블 목록 조회
        cursor.execute("""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table'
        """)
        tables = cursor.fetchall()
        
        # 각 테이블의 컬럼 정보 조회
        db_structure = {}
        for table in tables:
            table_name = table[0]
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            db_structure[table_name] = [col[1] for col in columns]
        
        conn.close()
        return db_structure
    except Exception as e:
        print(f"데이터베이스 구조 조회 실패: {str(e)}")
        return {}

def load_trading_decisions(days=7, conn=None, db_type='binance'):
    """거래 결정 기록 로드"""
    should_close = False
    if conn is None:
        conn = connect_db()
        should_close = True
    
    try:
        if db_type == 'binance':
            # 바이낸스 거래 기록 로드
            table_name = 'trading_decisions'
            select_columns = [
                'id', 'symbol', 'timestamp', 'price', 'volume_24h', 'volume_usdt_24h',
                'decision', 'analysis', 'technical_indicators', 'position_info',
                'risk_metrics', 'executed', 'pnl', 'created_at'
            ]
        else:
            # 업비트 거래 기록 로드
            table_name = 'trading_decisions'
            select_columns = [
                'id', 
                'timestamp', 
                'ticker as symbol',  # ticker를 symbol로 매핑
                'price',
                'volume_change as volume_24h',  # volume_change를 volume_24h로 매핑
                'profit_rate as pnl',  # profit_rate를 pnl로 매핑
                'decision',
                'position_ratio',
                'rsi',
                'ema_trend',
                'ha_pattern',
                'ai_response as analysis'  # ai_response를 analysis로 매핑
            ]
        
        # 테이블 존재 여부 확인
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """)
        if cursor.fetchone() is None:
            return pd.DataFrame()
        
        # 테이블 컬럼 조회
        columns = get_table_columns(conn, table_name)
        
        # 실제 존재하는 컬럼만 선택
        available_columns = [col.split(' as ')[0] for col in select_columns if col.split(' as ')[0] in columns]
        
        # SQL 쿼리 생성
        query = f"""
            SELECT {', '.join(select_columns)}
            FROM {table_name}
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
        
        if db_type == 'binance':
            # 바이낸스 JSON 컬럼 파싱
            for col in ['technical_indicators', 'position_info', 'risk_metrics']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if pd.notnull(x) and x else {})
        else:
            # 업비트 데이터 후처리
            df['price'] = df['price'].astype(float)
            if 'pnl' in df.columns:
                df['pnl'] = df['pnl'].astype(float) / 100  # 퍼센트를 소수로 변환
            if 'volume_24h' in df.columns:
                df['volume_24h'] = df['volume_24h'].astype(float)
            
            # 기술적 지표를 JSON 형식으로 변환
            df['technical_indicators'] = df.apply(
                lambda row: {
                    'rsi': row['rsi'] if 'rsi' in row else None,
                    'ema_trend': row['ema_trend'] if 'ema_trend' in row else None,
                    'ha_pattern': row['ha_pattern'] if 'ha_pattern' in row else None,
                    'position_ratio': row['position_ratio'] if 'position_ratio' in row else None
                },
                axis=1
            )
        
        return df
    except Exception as e:
        print(f"거래 기록 로드 중 오류: {str(e)}")
        return pd.DataFrame()
    finally:
        if should_close:
            conn.close()

def load_ai_analysis(days=7, conn=None):
    """AI 분석 로그 로드"""
    should_close = False
    if conn is None:
        conn = connect_db()
        should_close = True
    
    try:
        # 테이블 존재 여부 확인
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='ai_analysis_logs'
        """)
        if cursor.fetchone() is None:
            return pd.DataFrame()
        
        # 테이블 컬럼 조회
        columns = get_table_columns(conn, 'ai_analysis_logs')
        
        # 실제 존재하는 컬럼만 선택
        select_columns = [col for col in [
            'id', 'symbol', 'timestamp', 'prompt', 'response', 'created_at'
        ] if col in columns]
        
        query = f"""
            SELECT {', '.join(select_columns)}
            FROM ai_analysis_logs
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
        
        # symbol 컬럼이 없는 경우 기본값 추가
        if 'symbol' not in df.columns:
            df['symbol'] = 'BTCUSDT'  # 또는 적절한 기본값
        
        return df
    except Exception as e:
        print(f"AI 분석 로그 로드 중 오류: {str(e)}")
        return pd.DataFrame()
    finally:
        if should_close:
            conn.close()

def load_trading_reflections(days=7, conn=None):
    """반성일기 로드"""
    should_close = False
    if conn is None:
        conn = connect_db()
        should_close = True
    
    try:
        # 테이블 존재 여부 확인
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='trading_reflections'
        """)
        
        if cursor.fetchone() is None:
            return pd.DataFrame()
        
        query = """
            SELECT *
            FROM trading_reflections
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(f'-{days} days',))
        return df
    except Exception as e:
        print(f"반성일기 로드 중 오류: {str(e)}")
        return pd.DataFrame()
    finally:
        if should_close:
            conn.close()

def plot_price_chart(df, symbol):
    """가격 차트 생성"""
    fig = go.Figure()
    
    # 가격 라인
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        name='가격',
        line=dict(color='blue')
    ))
    
    # 거래 포인트 표시
    for decision in ['LONG', 'SHORT', 'CLOSE']:
        mask = df['decision'] == decision
        if mask.any():
            color = 'green' if decision == 'LONG' else 'red' if decision == 'SHORT' else 'gray'
            fig.add_trace(go.Scatter(
                x=df[mask]['timestamp'],
                y=df[mask]['price'],
                mode='markers',
                name=decision,
                marker=dict(
                    size=8,  # 마커 크기 축소
                    color=color,
                    symbol='triangle-up' if decision == 'LONG' else 'triangle-down' if decision == 'SHORT' else 'circle'
                )
            ))
    
    fig.update_layout(
        title=f'{symbol} 가격 차트',
        xaxis_title='시간',
        yaxis_title='가격',
        height=400,  # 높이 축소
        margin=dict(l=10, r=10, t=40, b=10),  # 마진 축소
        legend=dict(
            orientation="h",  # 범례를 가로로 변경
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # 모바일 터치 최적화
    fig.update_layout(
        dragmode='pan',  # 드래그로 이동
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def plot_pnl_chart(df):
    """손익 차트 생성"""
    fig = go.Figure()
    
    if 'pnl' in df.columns:
        # 적 손익 계산
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['cumulative_pnl'],
            name='누적 손익',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title='누적 손익 추이',
            xaxis_title='시간',
            yaxis_title='손익 (USDT)',
            height=400
        )
    else:
        # pnl 컬럼이 없을 때
        fig.update_layout(
            title='손익 데이터 없음',
            annotations=[{
                'text': '손익 데이터가 없습니다',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }],
            height=400
        )
    
    return fig

def plot_decision_pie(df):
    """거래 결정 분포 파이 차트"""
    if 'decision' in df.columns:
        decisions = df['decision'].value_counts()
        fig = px.pie(
            values=decisions.values,
            names=decisions.index,
            title='거래 결정 분포'
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title='거래 결정 데이터 없음',
            annotations=[{
                'text': '거래 결정 데이터가 없습니다',
                'xref': 'paper',
                'yref': 'paper',
                'showarrow': False,
                'font': {'size': 20}
            }]
        )
    return fig

def get_available_columns(df, desired_columns):
    """실제 존재하는 컬럼만 반환"""
    return [col for col in desired_columns if col in df.columns]

def calculate_statistics(df):
    """거래 통계 계산"""
    stats = {
        'total_trades': len(df),
        'total_pnl': df['pnl'].sum() if 'pnl' in df.columns else 0,
        'win_rate': 0,
        'avg_pnl': 0
    }
    
    if 'pnl' in df.columns and len(df) > 0:
        profitable_trades = len(df[df['pnl'] > 0])
        stats['win_rate'] = (profitable_trades / len(df)) * 100
        stats['avg_pnl'] = stats['total_pnl'] / len(df)
    
    return stats

def main():
    st.set_page_config(
        page_title="Trading Analysis",
        layout="wide",
        initial_sidebar_state="collapsed"  # 모바일에서는 사이드바 기본 숨김
    )
    
    # 모바일 최적화 CSS
    st.markdown("""
        <style>
        .stApp {
            max-width: 100%;
            padding: 1rem 1rem;
        }
        .st-emotion-cache-16idsys p {
            font-size: 14px;
        }
        .st-emotion-cache-16idsys {
            padding: 1rem 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("암호화폐 트레이딩 분석 대시보드")
    
    # 사용 가능한 데이터베이스 목록 가져오기
    available_dbs = get_available_databases()
    
    if not available_dbs:
        st.error("사용 가능한 데이터베이스가 없습니다.")
        return
    
    # 데이터베이스 선택
    selected_db = st.sidebar.selectbox(
        "거래소 선택",
        list(available_dbs.keys())
    )
    
    # 선택된 데이터베이스 경로
    db_path = available_dbs[selected_db]
    
    # 데이터베이스 구조 확인
    db_structure = get_database_info(db_path)
    if db_structure:
        st.sidebar.expander("데이터베이스 구조").write(db_structure)
    
    # 사이드바 설
    st.sidebar.header("설정")
    days = st.sidebar.slider("조회 기간 (일)", 1, 30, 7)
    auto_refresh = st.sidebar.checkbox("자동 새로고침", value=True)
    
    if auto_refresh:
        st.sidebar.info("60초마다 자동 새로고침됩니다")
        st.markdown(
            """
            <script>
                function reloadPage() {
                    setTimeout(function() {
                        window.location.reload();
                    }, 60000);
                }
                reloadPage();
            </script>
            """,
            unsafe_allow_html=True
        )
    
    # 데이터 로드 함수 수정
    def load_data(days):
        conn = connect_db(db_path)
        db_type = 'upbit' if 'upbit' in db_path else 'binance'
        trading_df = load_trading_decisions(days, conn, db_type)
        analysis_df = load_ai_analysis(days, conn)
        reflection_df = load_trading_reflections(days, conn)
        conn.close()
        return trading_df, analysis_df, reflection_df
    
    # 데이터 로드
    trading_df, analysis_df, reflection_df = load_data(days)
    
    # 데이터베이스 선택 옵션 수정 (반성일기 테이블이 없는 경우 제외)
    available_options = ["거래 기록 (trading_decisions)", "AI 분석 로그 (ai_analysis_logs)"]
    if not reflection_df.empty:
        available_options.append("반성일기 (trading_reflections)")
    available_options.append("통합 대시보드")
    
    db_selection = st.sidebar.radio(
        "데이터베이스 선택",
        available_options
    )
    
    if db_selection == "거래 기록 (trading_decisions)":
        st.header("거래 기록 데이터베이스")
        
        if len(trading_df) == 0:
            st.warning("해당 기간에 거래 기록이 없습니다.")
        else:
            # 코인별 필터
            symbols = trading_df['symbol'].unique()
            selected_symbol = st.selectbox("코인 선택", ['전체'] + list(symbols))
            
            if selected_symbol != '전체':
                filtered_df = trading_df[trading_df['symbol'] == selected_symbol]
                # 가격 차트 표시
                st.plotly_chart(plot_price_chart(filtered_df, selected_symbol), use_container_width=True)
            else:
                filtered_df = trading_df
            
            # 테이블 형식으로 데이터 표시
            st.subheader("거래 기록 테이블")
            # 표시할 컬럼 목록
            desired_columns = [
                'timestamp', 'symbol', 'price', 'decision', 
                'executed', 'pnl', 'created_at'
            ]
            # 실제 존재하는 컬럼만 선택
            available_columns = get_available_columns(filtered_df, desired_columns)
            display_df = filtered_df[available_columns].copy()
            st.dataframe(display_df)
            
            # 상세 기록
            st.subheader("상세 거래 기록")
            for _, row in filtered_df.iterrows():
                with st.expander(f"{row['timestamp']} - {row['symbol']} ({row['decision']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("기본 정보:")
                        st.write(f"가격: ${row['price']:,.2f}")
                        # 옵셔널 필드들은 존재할 경우에만 표시
                        if 'volume_usdt_24h' in row:
                            st.write(f"거래량(24h): ${row['volume_usdt_24h']:,.2f}")
                        if 'executed' in row:
                            st.write(f"실행 여부: {row['executed']}")
                        if 'pnl' in row:
                            st.write(f"손익: ${row['pnl']:,.2f}")
                    with col2:
                        if 'technical_indicators' in row:
                            st.write("기술적 지표:")
                            st.json(row['technical_indicators'])
                    if 'analysis' in row:
                        st.write("AI 분석:")
                        st.write(row['analysis'])
    
    elif db_selection == "AI 분석 로그 (ai_analysis_logs)":
        st.header("AI 분석 로그 데이터베이스")
        
        if len(analysis_df) == 0:
            st.warning("해당 기간에 AI 분석 로그가 없습니다.")
        else:
            # 코인별 필터
            symbols = analysis_df['symbol'].unique()
            selected_symbol = st.selectbox("코인 선택", ['전체'] + list(symbols))
            
            if selected_symbol != '전체':
                filtered_analysis = analysis_df[analysis_df['symbol'] == selected_symbol]
            else:
                filtered_analysis = analysis_df
            
            # 테이블 형식으로 기본 정보 표시
            st.subheader("AI 분석 로그 테이블")
            display_df = filtered_analysis[['timestamp', 'symbol', 'created_at']].copy()
            st.dataframe(display_df)
            
            # 상세 로그
            st.subheader("상세 AI 분석 로그")
            for _, row in filtered_analysis.iterrows():
                with st.expander(f"{row['timestamp']} - {row['symbol']}"):
                    st.subheader("프롬프트")
                    st.text(row['prompt'])
                    st.subheader("AI 응답")
                    st.text(row['response'])
    
    elif db_selection == "반성일기 (trading_reflections)":
        st.header("반성일기 데이터베이스")
        
        if len(reflection_df) == 0:
            st.warning("해당 기간에 반성일기가 없습니다.")
        else:
            # 테이블 형식으로 기본 정보 표시
            st.subheader("반성일기 목록")
            display_df = reflection_df[['timestamp', 'created_at']].copy()
            st.dataframe(display_df)
            
            # 세 반성일기
            st.subheader("상세 반성일기")
            for _, row in reflection_df.iterrows():
                with st.expander(f"{row['timestamp']}"):
                    st.markdown(row['reflection'])
    
    else:  # 통합 대시보드
        # 모바일에서는 탭 대신 섹션으로 표시
        st.header("거래 요약")
        if len(trading_df) == 0:
            st.warning("해당 기간에 거래 기록이 없습니다.")
        else:
            stats = calculate_statistics(trading_df)
            
            # 지표를 2열로 표시
            col1, col2 = st.columns(2)
            with col1:
                st.metric("총 거래 수", stats['total_trades'])
                st.metric("총 손익", f"${stats['total_pnl']:,.2f}")
            with col2:
                st.metric("승률", f"{stats['win_rate']:.1f}%")
                st.metric("평균 수익", f"${stats['avg_pnl']:,.2f}")
            
            # 차트를 세로로 배치
            st.plotly_chart(plot_decision_pie(trading_df), use_container_width=True)
            st.plotly_chart(plot_pnl_chart(trading_df), use_container_width=True)

if __name__ == "__main__":
    main()
