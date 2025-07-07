import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os

st.set_page_config(layout="wide")

# Detect Streamlit Cloud based on unique env variable
is_cloud = os.environ.get("SF_PARTNER") == "streamlit"

# --- SIDEBAR UI ---
with st.sidebar:
    st.markdown("## 🔧 App Controls")

    if is_cloud:
        reboot = st.button("🔁 Reboot App")
        st.caption(
            "Click to soft-reboot the app.\n\nThis will rerun the script and refresh any recent changes you've pushed to the repo. Useful after code updates or config changes.")

        if reboot:
            st.success("Rebooting app... Please wait.")
            st.rerun()
    else:
        st.info("This app is running locally.\n\nThe reboot button is only available on Streamlit Cloud.")

# st.write("Environment Variables:")
# st.code("\n".join(f"{k}={v}" for k, v in os.environ.items()))

# Function to create line chart data
def generate_line_data():
    epochs = list(range(1, 81))
    accuracy = np.linspace(0.75, 0.92, 80)
    return pd.DataFrame({'Epoch': epochs, 'Accuracy': accuracy})

# Function to create scatter plot data
def generate_scatter_data():
    return pd.DataFrame({
        'Simulated Prediction': np.random.rand(50) * 0.2 + 0.75,
        'Real Prediction': np.random.rand(50) * 0.2 + 0.75
    })

# Function to load experiment data for main-SingleSource and main-MultiSource
@st.cache_data
def load_experiment_data():
    csv_path = os.path.join(os.path.dirname(__file__), "csvs/gap_experiment_results.csv")
    df = pd.read_csv(csv_path)
    return df

# Function to read the contrastive domain log data (with caching)
@st.cache_data
def read_contrastive_log(path="contrastive_log.txt"):
    epochs = []
    accuracies = []
    with open(path, 'r') as f:
        for line in f:
            if "epoch" in line.lower() and "accuraccy" in line.lower():
                parts = line.strip().split(":")
                epoch = parts[0].split()[-1]
                acc = float(parts[-1])
                epochs.append(f"Epoch {epoch}")
                accuracies.append(acc)
    return pd.DataFrame({"Epoch": epochs, "Prediction Accuracy": accuracies})


# Function to display the contrastive domain chartAdd commentMore actions
def contrastive_domain_chart():
    # Fetching contrastive domain training log
    df_contrastive = read_contrastive_log()

    st.subheader("Contrastive Domain: Teacher Model Accuracy by Epoch")
    fig = px.bar(
        df_contrastive,
        x="Epoch",
        y="Prediction Accuracy",
        color="Epoch",
        text_auto=True,
        range_y=[0, 1],
        labels={"Epoch": "Training Epoch", "Prediction Accuracy": "Accuracy"},
        title="Teacher Model Accuracy during Contrastive Domain Training"
    )
    st.plotly_chart(fig, use_container_width=True)

# Function to load experiment data for main-performance-degradation
@st.cache_data
def load_accuracy_over_time_tab(chart_type):
    st.header("Accuracy Over Time with Domain Adaptation")
    csv_path = os.path.join(os.path.dirname(__file__), "csvs/accuracy_over_time.csv")

    try:
        df_accuracy = pd.read_csv(csv_path)

        # Ensure Days Passed is treated as categorical
        df_accuracy["Days Passed"] = df_accuracy["Days Passed"].astype(str)

        if chart_type == "Table":
            st.subheader("Accuracy Table")
            st.dataframe(df_accuracy)

        elif chart_type == "Line Chart":
            st.subheader("Line Chart of Accuracy Over Time")
            fig = px.line(df_accuracy, x="Days Passed", y="Accuracy", markers=True,
                          title="Accuracy changes over time with domain adaptation",
                          labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            fig.update_layout(
                xaxis=dict(type='category'),
                yaxis=dict(range=[0, 1]),
                xaxis_title="Days Passed",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig)

        elif chart_type == "Bar Chart":
            st.subheader("Bar Chart of Accuracy Over Time")
            fig = px.bar(df_accuracy, x="Days Passed", y="Accuracy",
                         title="Accuracy changes over time with domain adaptation",
                         labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            fig.update_layout(
                xaxis=dict(type='category'),
                yaxis=dict(range=[0, 1]),
                xaxis_title="Days Passed",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("Accuracy data not found. Please run the experiment script to generate accuracy_over_time.csv.")

# Function to load the meta accuracy CSV file
@st.cache_data
def load_meta_accuracy_csv(chart_type):
    try:
        df = pd.read_csv('csvs/meta_accuracy_over_time.csv')
        df["Days Passed"] = df["Days Passed"].astype(str)

        if chart_type == "Table":
            st.subheader("Meta Learning Accuracy Table")
            st.dataframe(df)

        elif chart_type == "Line Chart":
            st.subheader("Line Chart of Meta Learning Accuracy Over Time")
            fig = px.line(df, x="Days Passed", y="Accuracy", markers=True,
                          title="Meta Learning Accuracy Over Time",
                          labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            fig.update_layout(
                xaxis=dict(type='category'),
                yaxis=dict(range=[0, 1]),
                xaxis_title="Days Passed",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig)

        elif chart_type == "Bar Chart":
            st.subheader("Bar Chart of Meta Learning Accuracy Over Time")
            fig = px.bar(df, x="Days Passed", y="Accuracy",
                         title="Meta Learning Accuracy Over Time",
                         labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            fig.update_layout(
                xaxis=dict(type='category'),
                yaxis=dict(range=[0, 1]),
                xaxis_title="Days Passed",
                yaxis_title="Accuracy"
            )
            st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("CSV file not found. Please ensure the file 'meta_accuracy_over_time.csv' is in the correct location.")

# Store current active tab in session state
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Home"

# Function to simulate reboot of current tab
def reboot_tab(tab_name):
    st.session_state[f"{tab_name}_reboot"] = True

# Helper function to reset widgets (selections in multiselect boxes)
def reset_widget(*keys):
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Simulation-to-Reality Gap in Network Configuration", "Closing the Gap", "Runtime Adaptation"])

# Home Tab
with tab1:
    st.session_state.active_tab = "Home"

    if st.button("🔁 Reboot Home Tab"):
        reboot_tab("Home")
        st.rerun()

    if st.session_state.get("Home_reboot", False):
        st.success("Home tab has been reset!")
        st.session_state["Home_reboot"] = False

    st.header("**CAREER: Advancing Network Configuration and Runtime Adaptation Methods for Industrial Wireless Sensor-Actuator Networks**")

    st.subheader("**Team**")
    st.write('<p><strong>Primary Investigator</strong>: <a href="https://users.cs.fiu.edu/~msha/index.htm" target="_blank"> Mo Sha</a>, Associate Professor, Knight Foundation School of Computing and Information Sciences, Florida International University</p>', unsafe_allow_html=True)
    st.write("**PhD Student**: Aitian Ma")
    st.write("**Undergraduate Students**: Mario Casas, Jean Cruz")
    st.write("**Alumni**: Xia Cheng, Junyang Shi, Di Mu, Jean Tonday Rodriguez, Yitian Chen, Xingjian Chen")

    st.write("---")

    st.subheader("Project Period")
    st.write("3/12/2021 - 2/28/2026")
    st.image("NSF_logo.png", width=150)
    st.write(
        '<p>This project is sponsored by the National Science Foundation (NSF) through grant CNS-2150010 (replacing <a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2046538&HistoricalAwards=false" target="_blank">  CNS-2046538</a>) [<a href="https://www.nsf.gov/awardsearch/showAward?AWD_ID=2150010&HistoricalAwards=false" target="_blank">NSF award abstract</a>]</p>',
        unsafe_allow_html=True)

    st.write("---")

    st.subheader("**Project Abstract**")

    st.markdown("""
    A decade of real-world deployments of industrial wireless standards, 
    such as WirelessHART and ISA100, has demonstrated the feasibility of using IEEE 802.15.4-based wireless 
    sensor-actuator networks (WSANs) to achieve reliable and real-time wireless communication in industrial environments. 
    Although WSANs work satisfactorily most of the time thanks to years of research, 
    they are often difficult to configure as configuring a WSAN is a complex process, 
    which involves theoretical computation, simulation, field testing, among other tasks. 
    To support new services that require high data rates and mobile platforms, 
    industrial WSANs are adopting wireless technologies such as 5G and LoRa and becoming 
    increasingly hierarchical, heterogeneous, and complex, which significantly increases the network configuration difficulty. 
    This CAREER project aims to advance network configuration and runtime adaptation methods for industrial WSANs. 
    Research outcomes from this project will significantly enhance the resilience and agility of industrial WSANs 
    and reduce human involvement in network management, leading to a significant improvement in industrial efficiency and a 
    remarkable reduction of operating costs. By providing more advanced WSANs, the research outcomes from this project will 
    significantly spur the installation of WSANs in process industries and enable a broad range of new wireless-based applications, 
    which affects economics, security, and quality of life. This project enhances lectures and course project materials, 
    supports curriculum developments, creates research opportunities for undergraduate and graduate students, 
    and establishes outreach programs for K-12 students.
    
    Different from traditional methods that rely largely on experience and rules of thumb that involve a 
    coarse-grained analysis of network load or dynamics during a few field trials, this project develops a rigorous methodology 
    that leverages advanced machine learning techniques to configure and adapt WSANs by harvesting the valuable resources 
    (e.g., theoretical models and simulation methods) accumulated by the wireless research community. 
    This project develops new methods that leverage wireless simulations and deep learning to relate high-level network performance 
    to low-level network configurations and efficiently adapt the network at runtime to satisfy the performance requirements 
    specified by industrial applications. 
    This project demonstrates the performance of WSANs that are equipped with those new methods through 
    testbed experimentation, case study, and real-world validation. 
    The research outcomes from this project affects not only industrial WSANs but other complex wireless networks 
    as this project creates a replicable template for novel network configuration and runtime adaptation strategies 
    that advance the state of the art of wireless network management.
    """)

    st.write("---")

    st.subheader("**Publications**")

    st.write('<p><strong>[C]</strong> Aitian Ma, Jean Tonday Rodriguez, Mo Sha, and Dongsheng Luo, <a href="https://users.cs.fiu.edu/~msha/publications/dcoss2025.pdf" target="_blank"> Sensorless Air Temperature Sensing Using LoRa Link Characteristics</a>, IEEE International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT 2025), June 2025. </p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Aitian Ma, Jean Tonday Rodriguez, Mo Sha, and Dongsheng Luo, <a href="https://users.cs.fiu.edu/~msha/publications/MetroLivEnv25.pdf" target="_blank">  A LoRa-Based Energy-Harvesting Sensing System for Living Environment</a>, IEEE International Workshop on Metrology for Living Environment (MetroLivEnv), June 2025. </p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Aitian Ma and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/sac25.pdf" target="_blank"> WMN-CDA: Contrastive Domain Adaptation for Wireless Mesh Network Configuration</a>, ACM/SIGAPP Symposium On Applied Computing (SAC) Cyber-Physical Systems Track, March 2025, acceptance ratio: 5/21 = 23.8%. [<a href="https://github.com/aitianma/WMN-CDA" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Xia Cheng, Mo Sha, and Dong Chen, <a href="https://users.cs.fiu.edu/~msha/publications/ewsn2024.pdf" target="_blank"> Configuring Industrial Wireless Mesh Networks via Multi-Source Domain Adaptation</a>, ACM International Conference on Embedded Wireless Systems and Networks (EWSN), December 2024, acceptance ratio: 16/70 = 22.8%.</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Xia Cheng and Mo Sha, <a href="hhttps://users.cs.fiu.edu/~msha/publications/tosn2024.pdf" target="_blank"> MERA: Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, ACM Transactions on Sensor Networks, Vol. 20, Issue 4, pp. 97:1-97:24, July 2024. [<a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Junyang Shi, Aitian Ma, Xia Cheng, Mo Sha, and Xi Peng, <a href="https://users.cs.fiu.edu/~msha/publications/ton24.pdf target="_blank"> Adapting Wireless Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, IEEE/ACM Transactions on Networking, Vol. 32, Issue 3, pp. 1983-1998, June 2024. [<a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Aitian Ma, Jean Tonday Rodriguez, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icccn24.pdf" target="_blank"> Enabling Reliable Environmental Sensing with LoRa, Energy Harvesting, and Domain Adaptation</a>, IEEE International Conference on Computer Communications and Networks (ICCCN), July 2024, acceptance ratio: 47/157 = 29.9%.</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Xu Zheng, Tianchun Wang, Wei Cheng, Aitian Ma, Haifeng Chen, Mo Sha, and Dongsheng Luo, <a href="https://users.cs.fiu.edu/~msha/publications/iclr24.pdf" target="_blank"> Parametric Augmentation for Time Series Contrastive Learning</a>, International Conference on Learning Representations (ICLR), May 2024, acceptance ratio: 2260/7262 = 31.1%.</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Xia Cheng, Junyang Shi, and Mo Sha, and Linke Guo, <a href="https://users.cs.fiu.edu/~msha/publications/ton23.pdf" target="_blank"> Revealing Smart Selective Jamming Attacks in WirelessHART Networks</a>, IEEE/ACM Transactions on Networking, Vol. 31, Issue 4, pp. 1611-1625, August, 2023. [<a href="https://users.cs.fiu.edu/~msha/publications/ton23.pdf" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Xia Cheng and Mo Sha, <a href= "https://users.cs.fiu.edu/~msha/publications/iwqos23.pdf" target="_blank"> Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, IEEE/ACM International Symposium on Quality of Service (IWQoS), June 2023, acceptance ratio: 62/264 = 23.5%. [<a href="https://github.com/iiot-research/Selective-Jamming" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Di Mu, Yitian Chen, Xingjian Chen, Junyang Shi, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/infocom23.pdf" target="_blank"> Enabling Direct Message Dissemination in Industrial Wireless Networks via Cross-Technology Communication</a>, IEEE International Conference on Computer Communications (INFOCOM), May 2023, acceptance ratio: 252/1312 = 19.2%. [<a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Xia Cheng and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn2022_2.pdf" target="_blank"> Autonomous Traffic-Aware Scheduling for Industrial Wireless Sensor-Actuator Networks</a>, ACM Transactions on Sensor Networks, Vol. 19, Issue 2, pp. 38:1-38:25, February 2023. [<a href="https://github.com/iiot-research/Autonomous-Scheduling" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Qi Li, Keyang Yu, Dong Chen, Mo Sha, and Long Cheng, <a href="https://users.cs.fiu.edu/~msha/publications/cns22.pdf" target="_blank"> TrafficSpy: Disaggregating VPN-encrypted IoT Network Traffic for User Privacy Inference</a>, IEEE Conference on Communications and Network Security (CNS), October 2022, acceptance ratio: 43/122 = 35.2%. </p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Junyang Shi and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icccn22.pdf" target="_blank"> Localizing Campus Shuttles from One Single Base Station Using LoRa Link Characteristics</a>, IEEE International Conference on Computer Communications and Networks (ICCCN), July 2022, acceptance ratio: 39/130=30.0%. [<a href="https://github.com/junyang28/lorashuttlebus" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Junyang Shi, Xingjian Chen, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn2022.pdf" target="_blank"> Enabling Cross-technology Communication from LoRa to ZigBee in the 2.4 GHz Band</a>, ACM Transactions on Sensor Networks, Vol. 18, Issue 2, pp. 21:1-21:23, May 2022. [<a href="https://github.com/junyang28/paper-ctclora" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[J]</strong> Junyang Shi, Di Mu, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn21.pdf" target="_blank"> Enabling Cross-technology Communication from LoRa to ZigBee via Payload Encoding in Sub-1 GHz Bands</a>, ACM Transactions on Sensor Networks, Vol. 18, Issue 1, pp. 6:1-6:26, February 2022. [<a href="https://github.com/junyang28/paper-lorabee" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Xiao Cheng and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icnp21.pdf" target="_blank">  ATRIA: Autonomous Traffic-Aware Transmission Scheduling for Industrial Wireless Sensor-Actuator Networks</a>, IEEE International Conference on Network Protocols (ICNP), November 2021, acceptance ratio: 38/154 = 24.6%. [<a href="https://github.com/iiot-research/Autonomous-Scheduling" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p><strong>[C]</strong> Junyang Shi, Mo Sha, and Xi Peng, <a href="https://users.cs.fiu.edu/~msha/publications/nsdi21.pdf" target="_blank"> Adapting Wireless Mesh Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, USENIX Symposium on Networked Systems Design and Implementation (NSDI), April 2021, acceptance ratio (fall deadline): 40/255 = 15.6%. [<a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)

# Gap Analysis Tab
with tab2:

    st.session_state.active_tab = "Simulation-to-Reality Gap in Network Configuration"

    # Reboot button logic
    if st.button("🔁 Reboot Gap Analysis Tab"):
        # Clear relevant session state before rerun
        st.session_state["Simulation-to-Reality Gap in Network Configuration_reboot"] = True
        st.session_state["gap_reboot_flag"] = True # <- flag to trigger reset on next run
        st.rerun()

    # Reset message
    if st.session_state.get("gap_reboot_flag"):
        st.success("Simulation-to-Reality Gap tab has been reset!")
        # Fully reset the widget so it appears empty
        reset_widget("gap_analysis_selected_visuals")
        st.session_state["gap_reboot_flag"] = False # Clear the flag

    st.header("Simulation-to-Reality Gap in Network Configuration")

    # --- Initialize session state for multiselect ---
    if "gap_analysis_selected_visuals" not in st.session_state:
        st.session_state["gap_analysis_selected_visuals"] = []

    # Load experiment data from CSV
    gap_df = load_experiment_data()

    # Handle mapping if the CSV uses short domain codes
    if 'Domain Setting' in gap_df.columns:
        label_map = {
            'Dˢ → Dˢ': 'Train: Simulation, Test: Simulation (Dˢ → Dˢ)',
            'Dˢ → Dᵖ': 'Train: Simulation, Test: Physical (Dˢ → Dᵖ)',
            'Dᵖ → Dᵖ': 'Train: Physical, Test: Physical (Dᵖ → Dᵖ)'
        }
        gap_df['Training and testing on different data sets'] = gap_df['Domain Setting'].map(label_map)
    elif 'Training and testing on different data sets' not in gap_df.columns:
        st.error("The loaded CSV must contain either 'Domain Setting' or 'Training and testing on different data sets'.")
        st.stop()

    # Rename for chart consistency
    if 'Accuracy' in gap_df.columns:
        gap_df = gap_df.rename(columns={'Accuracy': 'Prediction Accuracy'})

    st.write('<p>Junyang Shi, Mo Sha, and Xi Peng, '
             '<a href="https://users.cs.fiu.edu/~msha/publications/nsdi21.pdf" target="_blank">'
             'Adapting Wireless Mesh Network Configuration '
             'from Simulation to Reality via Deep Learning based Domain Adaptation</a>, '
             'USENIX Symposium on Networked Systems Design and Implementation (NSDI), April 2021. '
             '[<a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">'
             'source code and data</a>]</p>', unsafe_allow_html=True)

    st.subheader("Select Visualizations to Display:")
    gap_analysis_selected_visuals = st.multiselect(
        "Choose one or more visualizations",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Data Table"],
        key="gap_analysis_selected_visuals"
    )

    if gap_analysis_selected_visuals:
        if "Bar Chart" in gap_analysis_selected_visuals:
            fig = px.bar(
                gap_df,
                x='Training and testing on different data sets',
                y='Prediction Accuracy',
                title="Simulation-to-Reality Gap",
                labels={
                    'Training and testing on different data sets': 'Training and testing on different data sets',
                    'Prediction Accuracy': 'Prediction Accuracy'
                },
                color='Training and testing on different data sets',
                color_discrete_sequence=['#A9A9A9', '#87CEEB', '#FFD700']
            )
            fig.update_layout(
                bargap=0.4,
                xaxis={'categoryorder': 'array', 'categoryarray': gap_df['Training and testing on different data sets'].tolist()}
            )
            st.plotly_chart(fig)

        if "Line Chart" in gap_analysis_selected_visuals:
            line_data = generate_line_data()
            fig = px.line(
                line_data,
                x='Epoch',
                y='Accuracy',
                title="Accuracy Progression",
                labels={'Accuracy': 'Prediction Accuracy'}
            )
            st.plotly_chart(fig)

        if "Scatter Plot" in gap_analysis_selected_visuals:
            scatter_data = generate_scatter_data()
            fig = px.scatter(
                scatter_data,
                x='Simulated Prediction',
                y='Real Prediction',
                title="Simulated vs Real Predictions",
                trendline="ols",
                labels={'Simulated Prediction': 'Simulated Accuracy', 'Real Prediction': 'Real Accuracy'}
            )
            st.plotly_chart(fig)

        if "Data Table" in gap_analysis_selected_visuals:
            st.subheader("Gap Analysis Table")
            st.dataframe(gap_df[['Training and testing on different data sets', 'Prediction Accuracy']])
    else:
        st.info("No data selected to display.")

with tab3:

    st.session_state.active_tab = "Closing the Gap"

    # Reboot button logic
    if st.button("🔁 Reboot Closing the Gap Tab"):
        # Clear relevant session state before rerun
        st.session_state["Closing the Gap_reboot"] = True
        st.session_state["closing_reboot_flag"] = True # <- flag to trigger reset on next run
        st.rerun()

    # Reset message
    if st.session_state.get("closing_reboot_flag"):
        st.success("Closing the Gap tab has been reset!")
        # Fully reset the widget so it appears empty
        reset_widget("single_source", "contrastive_domain", "multi_source", "combined_source")
        st.session_state["closing_reboot_flag"] = False # Clear the flag

    # --- Initialize session state for multiselect ---
    if "single_source" not in st.session_state:
        st.session_state["single_source"] = []
    if "contrastive_domain" not in st.session_state:
        st.session_state["contrastive_domain"] = []
    if "multi_source" not in st.session_state:
        st.session_state["multi_source"] = []
    if "combined_source" not in st.session_state:
        st.session_state["combined_source"] = []

    st.header("Our solutions to close the simulation-to-reality gap in network configuration.")

    # Load precomputed data
    data = load_experiment_data()

    if data is not None and not data.empty:
        # Papers for domain adaptation
        st.write("(1) Leveraging domain adaptation to close the gap.")
        st.write('<p> Junyang Shi, Mo Sha, and Xi Peng, '
                 '<a href="https://users.cs.fiu.edu/~msha/publications/nsdi21.pdf" target="_blank">Adapting Wireless Mesh Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, '
                 'USENIX Symposium on Networked Systems Design and Implementation (NSDI), April 2021. '
                 '[<a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)

        st.write('<p> Junyang Shi, Aitian Ma, Xia Cheng, Mo Sha, and Xi Peng, '
                 '<a href="https://users.cs.fiu.edu/~msha/publications/ton24.pdf" target="_blank">Adapting Wireless Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, '
                 'IEEE/ACM Transactions on Networking, Vol. 32, Issue 3, pp. 1983-1998, June 2024. '
                 '[<a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">source code and data</a>]</p>', unsafe_allow_html=True)

        # 🟦 Single Source Visuals
        single_source_visuals = st.multiselect(
            "Choose NSDI visualizations",
            ["Bar Chart", "Line Chart", "Table"],
            key="single_source"
        )

        # Load single source CSV
        single_source_data = pd.read_csv("csvs/single_source_accuracy.csv")

        if single_source_visuals:
            if "Bar Chart" in single_source_visuals:
                fig = px.bar(single_source_data, x='Shots', y='Single Source Accuracy',
                             title="NSDI Prediction Accuracy (Bar)",
                             labels={
                                 'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                 'Single Source Accuracy': 'Prediction Accuracy'
                             },
                             range_y=[0, 1])
                st.plotly_chart(fig)

            if "Line Chart" in single_source_visuals:
                fig = px.line(single_source_data, x='Shots', y='Single Source Accuracy', markers=True,
                              title="NSDI Prediction Accuracy (Line)",
                              labels={
                                  'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                  'Single Source Accuracy': 'Prediction Accuracy'
                              })
                st.plotly_chart(fig)

            if "Table" in single_source_visuals:
                st.subheader("NSDI Source Table")
                st.dataframe(single_source_data[['Shots', 'Single Source Accuracy']])

        st.write("(2) Using contrastive domain adaptation to close the gap.")
        st.write('<p> Aitian Ma and Mo Sha, '
                 '<a href="https://users.cs.fiu.edu/~msha/publications/sac25.pdf" target="_blank">'
                 'WMN-CDA: Contrastive Domain Adaptation for Wireless Mesh Network Configuration</a>, '
                 'ACM/SIGAPP Symposium On Applied Computing (SAC) Cyber-Physical Systems Track, March 2025</p>', unsafe_allow_html=True)

        # 🟪 Contrastive Domain Visuals
        contrastive_domain_visuals = st.multiselect(
            "Choose SAC visualizations",
            ["Bar Chart", "Line Chart", "Table"],
            key="contrastive_domain"
        )

        # Load CSV data for SAC
        contrastive_data = pd.read_csv("csvs/contrastive_domain_accuracy.csv")

        if contrastive_domain_visuals:
            if "Bar Chart" in contrastive_domain_visuals:
                contrastive_melted = contrastive_data.melt(
                    id_vars='Shots', var_name='Method', value_name='Prediction Accuracy'
                )
                fig = px.bar(
                    contrastive_melted, x='Shots', y='Prediction Accuracy', color='Method',
                    barmode='group',
                    title="[SAC] Contrastive Domain Adaptation Accuracy (Bar)",
                    labels={
                        'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                        'Prediction Accuracy': 'Prediction Accuracy'
                    },
                    range_y=[0, 1]
                )
                fig.update_layout(
                    xaxis=dict(type='category'),
                    xaxis_title='Amount of Physical Data Used for Training (No. of Shots)',
                    yaxis_title='Prediction Accuracy'
                )
                st.plotly_chart(fig)

            if "Line Chart" in contrastive_domain_visuals:
                fig = px.line(contrastive_data, x='Shots', y='CL-OMNet', markers=True,
                              title="[SAC] Contrastive Domain Adaptation Accuracy (Line)",
                              labels={
                                  'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                  'CL-OMNet': 'Prediction Accuracy'
                              })
                st.plotly_chart(fig)

            if "Table" in contrastive_domain_visuals:
                st.subheader("Contrastive Domain Results (SAC - OMNeT)")
                st.dataframe(contrastive_data)

        st.write("(3) Employing multi-source domain adaptation to close the gap.")
        st.write('<p>Xia Cheng, Mo Sha, and Dong Chen, '
                 '<a href="https://users.cs.fiu.edu/~msha/publications/ewsn2024.pdf" target="_blank">Configuring Industrial Wireless Mesh Networks via Multi-Source Domain Adaptation</a>, '
                 'ACM International Conference on Embedded Wireless Systems and Networks (EWSN), December 2024</p>', unsafe_allow_html=True)

        # 🟩 Multi Source Visuals
        multi_source_visuals = st.multiselect(
            "Choose EWSN visualizations",
            ["Bar Chart", "Line Chart", "Table"],
            key="multi_source"
        )

        # Load data
        multi_source_data = pd.read_csv("csvs/multi_source_accuracy.csv")

        if multi_source_visuals:
            if "Bar Chart" in multi_source_visuals:
                fig = px.bar(multi_source_data, x='Shots', y='Multi Source Accuracy',
                             title="EWSN Prediction Accuracy (Bar)",
                             labels={
                                 'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                 'Multi Source Accuracy': 'Prediction Accuracy'
                             },
                             range_y=[0, 1])
                st.plotly_chart(fig)

            if "Line Chart" in multi_source_visuals:
                fig = px.line(multi_source_data, x='Shots', y='Multi Source Accuracy', markers=True,
                              title="EWSN Prediction Accuracy (Line)",
                              labels={
                                  'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                  'Multi Source Accuracy': 'Prediction Accuracy'
                              })
                st.plotly_chart(fig)

            if "Table" in multi_source_visuals:
                st.subheader("Multi Source Table")
                st.dataframe(multi_source_data[['Shots', 'Multi Source Accuracy']])

        # 🟪 Combined Source Visualizations
        combined_source_selected_visuals = st.multiselect(
            "**See the comparisons among different solutions.**",
            [
                "Combined Bar Chart", "Combined Line Chart", "Combined Table",
            ], key="combined_source"
        )

        single_source_data.rename(columns={'Accuracy': 'Single Source Accuracy'}, inplace=True)
        multi_source_data.rename(columns={'Accuracy': 'Multi Source Accuracy'}, inplace=True)
        contrastive_data.rename(columns={'Accuracy': 'CL-OMNet'}, inplace=True)

        # Step 1: Merge the datasets on 'Shots'
        merged_data = pd.merge(single_source_data, multi_source_data, on='Shots', how='outer')
        merged_data = pd.merge(merged_data, contrastive_data, on='Shots', how='outer')

        # Step 2: Reshape the data into a long format for easy plotting
        combined_melted = merged_data.melt(
            id_vars=['Shots'],  # Keep 'Shots' as the id
            value_vars=['Single Source Accuracy', 'Multi Source Accuracy', 'CL-OMNet'],
            # Columns to melt
            var_name='Source Type',  # New column for the source types
            value_name='Prediction Accuracy'  # The actual values for the plot
        )

        # Combined Source Visualizations
        if combined_source_selected_visuals:
            if "Combined Bar Chart" in combined_source_selected_visuals:
                # Step 1: Merge datasets
                merged_data = pd.merge(single_source_data, multi_source_data, on='Shots', how='outer')
                merged_data = pd.merge(merged_data, contrastive_data, on='Shots', how='outer')

                # Adjust labels for clarity
                combined_melted['Source Type'] = combined_melted['Source Type'].map({
                    'Single Source Accuracy': 'Single Source',
                    'Multi Source Accuracy': 'Multi Source',
                    'CL-OMNet': 'CL-OMNet'
                })

                # Grouped Bar Chart
                fig = px.bar(combined_melted, x='Shots', y='Prediction Accuracy',
                             color='Source Type', barmode='group',
                             title="Prediction Accuracy Comparison (Bar)",
                             labels={
                                 'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                 'Prediction Accuracy': 'Prediction Accuracy'
                             },
                             range_y=[0, 1])

                # Set the x-axis as categorical to avoid unexpected continuous behavior
                fig.update_layout(
                    xaxis=dict(type='category'),  # Treat x-axis as categorical
                    xaxis_title='Amount of Physical Data Used for Training (No. of Shots)',
                    yaxis_title='Prediction Accuracy'
                )

                st.plotly_chart(fig)

            if "Combined Line Chart" in combined_source_selected_visuals:
                # Line Chart
                fig = px.line(combined_melted, x='Shots', y='Prediction Accuracy',
                              color='Source Type', markers=True,
                              title="Prediction Accuracy Comparison (Line)",
                              labels={
                                  'Shots': 'Amount of Physical Data Used for Training (No. of Shots)',
                                  'Prediction Accuracy': 'Prediction Accuracy'
                              })

                # Set the x-axis as categorical to avoid unexpected continuous behavior
                fig.update_layout(
                    xaxis=dict(type='category'),  # Treat x-axis as categorical
                    xaxis_title='Amount of Physical Data Used for Training (No. of Shots)',
                    yaxis_title='Prediction Accuracy'
                )

                st.plotly_chart(fig)

            if "Combined Table" in combined_source_selected_visuals:

                # Step 2: Create separate columns for each source type
                combined_data_table = merged_data[
                    ['Shots', 'Single Source Accuracy', 'Multi Source Accuracy', 'CL-OMNet']]

                # Rename the columns for clarity
                combined_data_table = combined_data_table.rename(columns={
                    'Single Source Accuracy': 'Single Source Accuracy',
                    'Multi Source Accuracy': 'Multi Source Accuracy',
                    'Contrastive Learning Accuracy': 'CL-OMNet'
                })

                # Display the table with separate columns
                st.subheader("Combined Source Table")
                st.dataframe(combined_data_table)
        else:
            st.info("No data selected to display in Combined Source Results.")
    else:
        st.error("No valid data available to display.")

# Meta Learning Tab
with tab4:
    st.session_state.active_tab = "Runtime Adaptation"

    # Reboot button logic
    if st.button("🔁 Reboot Runtime Adaptation Tab"):
        # Clear relevant session state before rerun
        st.session_state["Runtime Adaptation_reboot"] = True
        st.session_state["domain_reboot_flag"] = True  # <- flag to trigger reset on next run
        st.rerun()

    # Reset message
    if st.session_state.get("domain_reboot_flag"):
        st.success("Runtime Adaptation tab has been reset!")
        # Fully reset the widget so it appears empty
        reset_widget("meta_learning", "domain_adaptation")
        st.session_state["domain_reboot_flag"] = False  # Clear the flag

    # --- Initialize session state for multiselect ---
    if "meta_learning" not in st.session_state:
        st.session_state["meta_learning"] = []
    if "domain_adaptation" not in st.session_state:
        st.session_state["domain_adaptation"] = []

    st.header("Meta Learning Results")

    st.write("Using domain adaptation to adapt the network configuration at runtime.")
    st.write('<p> Xia Cheng and Mo Sha, '
             '<a href="https://users.cs.fiu.edu/~msha/publications/iwqos23.pdf" target="_blank">'
             'Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, '
             'IEEE/ACM International Symposium on Quality of Service (IWQoS), June 2023. '
             '<a href="https://github.com/iiot-research/Selective-Jamming" target="_blank">'
             '[source code and data]</a></p>', unsafe_allow_html=True)

    st.write('<p> Xia Cheng and Mo Sha, '
             '<a href="https://users.cs.fiu.edu/~msha/publications/tosn2024.pdf" target="_blank">'
             'MERA: Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, '
             'ACM Transactions on Sensor Networks, Vol. 20, Issue 4, pp. 97:1-97:24, July 2024. '
             '<a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">'
             '[source code and data]</a></p>', unsafe_allow_html=True)

    st.subheader("**Prediction accuracy decreases over time. More details can be found below:**")
    st.write('Xia Cheng and Mo Sha, '
             '<a href="https://users.cs.fiu.edu/~msha/publications/iwqos23.pdf" target="_blank">'
             'Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, '
             'IEEE/ACM International Symposium on Quality of Service (IWQoS), June 2023. '
             '<a href="https://github.com/iiot-research/Selective-Jamming" target="_blank">'
             '[source code and data]</a></p>', unsafe_allow_html=True)

    domain_adaptation_results_tab4 = st.multiselect(
        "Choose visualizations to display for Domain Adaptation Results",
        ["Bar Chart", "Line Chart", "Data Table"],
        key="domain_adaptation"
    )
    if domain_adaptation_results_tab4:
        if "Bar Chart" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Bar Chart")

        if "Line Chart" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Line Chart")

        if "Data Table" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Table")
    else:
        st.info("No data selected to display Domain Adaptation Results.")

    st.subheader("**Using domain adaptation to adapt the network configuration at runtime.**")
    st.write('<p>Xia Cheng and Mo Sha, '
             '<a href="https://users.cs.fiu.edu/~msha/publications/iwqos23.pdf" target="_blank">'
             'Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, '
             'IEEE/ACM International Symposium on Quality of Service (IWQoS), June 2023. '
             '[<a href="https://github.com/iiot-research/Selective-Jamming" target="_blank">'
             'source code and data</a>]</p>', unsafe_allow_html=True)
    st.write('<p>Xia Cheng and Mo Sha, '
             '<a href="hhttps://users.cs.fiu.edu/~msha/publications/tosn2024.pdf" target="_blank">'
             'MERA: Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, '
             'ACM Transactions on Sensor Networks, Vol. 20, Issue 4, pp. 97:1-97:24, July 2024. '
             '[<a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">'
             'source code and data</a>]</p>', unsafe_allow_html=True)

    meta_learning_selected_visuals = st.multiselect(
        "Choose visualizations to display for Meta Learning Results",
        ["Bar Chart", "Line Chart", "Data Table"],
        key="meta_learning"
    )

    if meta_learning_selected_visuals:
        # Bar Chart
        if "Bar Chart" in meta_learning_selected_visuals:
            load_meta_accuracy_csv(chart_type="Bar Chart")

        # Line Chart
        if "Line Chart" in meta_learning_selected_visuals:
            load_meta_accuracy_csv(chart_type="Line Chart")

        # Data Table
        if "Data Table" in meta_learning_selected_visuals:
            load_meta_accuracy_csv(chart_type="Table")
    else:
        st.info("No data selected to display in Meta Learning Results.")
