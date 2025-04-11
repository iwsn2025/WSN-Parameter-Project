import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import os

# Set the page layout to wide
st.set_page_config(layout="wide")

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

# Function to load experiment data from the CSV file for main-SingleSource and main-MultiSource
@st.cache_data
def load_experiment_data():
    csv_path = os.path.join(os.path.dirname(__file__), "experiment_results.csv")

    # Check if the file exists
    if not os.path.exists(csv_path):
        st.error(f"File {csv_path} not found. Please ensure the file exists in the correct directory.")
        return None  # Return None if the file is not found

    try:
        # Try reading the CSV file
        data = pd.read_csv(csv_path)

        # Check if necessary columns are in the CSV file
        expected_columns = ['Shots', 'Single Source Accuracy', 'Multi Source Accuracy']
        if not all(col in data.columns for col in expected_columns):
            st.error(f"The CSV file is missing one or more expected columns: {', '.join(expected_columns)}")
            return None

        return data  # Return the loaded data if everything is correct

    except Exception as e:
        # Catch any errors and display them
        st.error(f"Error loading CSV: {e}")
        return None

# Function to load experiment data for main-performance-degradation
@st.cache_data
def load_accuracy_over_time_tab(chart_type):
    st.header("Accuracy Over Time with Domain Adaptation")
    csv_path = os.path.join(os.path.dirname(__file__), "accuracy_over_time.csv")

    try:
        df_accuracy = pd.read_csv(csv_path)

        if chart_type == "Table":
            st.subheader("Accuracy Table")
            st.dataframe(df_accuracy)

        elif chart_type == "Line Chart":
            st.subheader("Line Chart of Accuracy Over Time")
            fig = px.line(df_accuracy, x="Days Passed", y="Accuracy", markers=True,
                          title="Accuracy changes over time with domain adaptation",
                          labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            st.plotly_chart(fig)

        elif chart_type == "Bar Chart":
            st.subheader("Bar Chart of Accuracy Over Time")
            fig = px.bar(df_accuracy, x="Days Passed", y="Accuracy",
                         title="Accuracy changes over time with domain adaptation",
                         labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("Accuracy data not found. Please run the experiment script to generate accuracy_over_time.csv.")


# Function to load the meta accuracy CSV file
@st.cache_data
def load_meta_accuracy_csv(chart_type):
    try:
        # Read the CSV file
        df = pd.read_csv('meta_accuracy_over_time.csv')

        if chart_type == "Table":
            st.subheader("Meta Learning Accuracy Table")
            st.dataframe(df)

        elif chart_type == "Line Chart":
            st.subheader("Line Chart of Meta Learning Accuracy Over Time")
            st.line_chart(df.set_index('Days Passed')['Accuracy'])

        elif chart_type == "Bar Chart":
            st.subheader("Bar Chart of Meta Learning Accuracy Over Time")
            fig = px.bar(df, x="Days Passed", y="Accuracy",
                         title="Meta Learning Accuracy Over Time",
                         labels={"Days Passed": "Days Passed", "Accuracy": "Accuracy"})
            st.plotly_chart(fig)

    except FileNotFoundError:
        st.error("CSV file not found. Please ensure the file 'meta_accuracy_over_time.csv' is in the correct location.")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Home", "Gap Analysis", "Single vs Multi-Source", "Meta Learning"])

# Home Tab
with tab1:
    st.header("Welcome to the Wireless Sensor-Actuator Networks (WSANs) Visualization Hub")

    st.write("""
    Welcome to the cutting-edge platform designed to revolutionize the configuration and adaptation of industrial Wireless Sensor-Actuator Networks (WSANs). Building on a decade of research and real-world deployments of industrial wireless standards like WirelessHART and ISA100, this platform empowers you to visualize and interact with complex data in an intuitive, user-friendly way.

    Our goal is simple: to make industrial WSANs more resilient, efficient, and easier to manage. By harnessing advanced machine learning, deep learning, and wireless simulations, we are pushing the boundaries of network configuration and runtime adaptation. This website provides a powerful toolset to visualize these intricate networks in real-time, helping users understand, optimize, and adapt their systems with ease.

    Whether you’re a researcher, an industry professional, or someone looking to explore the future of wireless networks, our platform offers a dynamic way to interact with your data. Key features include:

    - **Real-time Visualization**: View the performance of industrial WSANs and how they adapt in real-time to changing conditions.
    - **Data Exploration**: Dive into data with interactive charts, graphs, and tables that help you uncover insights from complex simulations and real-world testbeds.
    - **Advanced Network Configuration**: Understand and optimize network configurations using deep learning-powered analysis and simulation results.
    - **Seamless Integration**: Access and compare data across multiple sources with intuitive charts and tables, all designed for ease of use.

    This project aims to reduce human involvement in network management, enhance industrial efficiency, and significantly lower operational costs. By visualizing these powerful techniques, we’re not just advancing WSANs but also making them more accessible to industries that rely on real-time, high-performance wireless networks.

    Explore our tools, visualize your data, and be part of the next generation of wireless network management!
    """)

    st.subheader("Publications")

    # Publications in a 3-column layout with dividers
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write('<p><strong>[C]</strong> Aitian Ma, Jean Tonday Rodriguez, Mo Sha, and Dongsheng Luo, Sensorless Air Temperature Sensing Using LoRa Link Characteristics, IEEE International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT 2025), June 2025. </p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Aitian Ma and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/sac25.pdf" target="_blank"> WMN-CDA: Contrastive Domain Adaptation for Wireless Mesh Network Configuration</a>, ACM/SIGAPP Symposium On Applied Computing (SAC) Cyber-Physical Systems Track, March 2025, acceptance ratio: 5/21 = 23.8%.</p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Xia Cheng, Mo Sha, and Dong Chen, <a href="https://users.cs.fiu.edu/~msha/publications/ewsn2024.pdf" target="_blank"> Configuring Industrial Wireless Mesh Networks via Multi-Source Domain Adaptation</a>, ACM International Conference on Embedded Wireless Systems and Networks (EWSN), December 2024.</p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Xia Cheng and Mo Sha, <a href="hhttps://users.cs.fiu.edu/~msha/publications/tosn2024.pdf" target="_blank"> MERA: Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, ACM Transactions on Sensor Networks, Vol. 20, Issue 4, pp. 97:1-97:24, July 2024. <a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Junyang Shi, Aitian Ma, Xia Cheng, Mo Sha, and Xi Peng, <a href="https://users.cs.fiu.edu/~msha/publications/ton24.pdf target="_blank"> Adapting Wireless Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, IEEE/ACM Transactions on Networking, Vol. 32, Issue 3, pp. 1983-1998, June 2024. <a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)

    with col2:
        st.write('<p><strong>[C]</strong> Aitian Ma, Jean Tonday Rodriguez, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icccn24.pdf" target="_blank"> Enabling Reliable Environmental Sensing with LoRa, Energy Harvesting, and Domain Adaptation</a>, IEEE International Conference on Computer Communications and Networks (ICCCN), July 2024, acceptance ratio: 47/157 = 29.9%.</p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Xu Zheng, Tianchun Wang, Wei Cheng, Aitian Ma, Haifeng Chen, Mo Sha, and Dongsheng Luo, <a href="https://users.cs.fiu.edu/~msha/publications/iclr24.pdf" target="_blank"> Parametric Augmentation for Time Series Contrastive Learning</a>, International Conference on Learning Representations (ICLR), May 2024, acceptance ratio: 2260/7262 = 31.1%.</p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Xia Cheng, Junyang Shi, and Mo Sha, and Linke Guo, <a href="https://users.cs.fiu.edu/~msha/publications/ton23.pdf" target="_blank"> Revealing Smart Selective Jamming Attacks in WirelessHART Networks</a>, IEEE/ACM Transactions on Networking, Vol. 31, Issue 4, pp. 1611-1625, August, 2023. <a href="https://users.cs.fiu.edu/~msha/publications/ton23.pdf" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Xia Cheng and Mo Sha, <a href= "https://users.cs.fiu.edu/~msha/publications/iwqos23.pdf" target="_blank"> Meta-Learning Based Runtime Adaptation for Industrial Wireless Sensor-Actuator Networks</a>, IEEE/ACM International Symposium on Quality of Service (IWQoS), June 2023, acceptance ratio: 62/264 = 23.5%. <a href="https://github.com/iiot-research/Selective-Jamming" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Di Mu, Yitian Chen, Xingjian Chen, Junyang Shi, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/infocom23.pdf" target="_blank"> Enabling Direct Message Dissemination in Industrial Wireless Networks via Cross-Technology Communication</a>, IEEE International Conference on Computer Communications (INFOCOM), May 2023, acceptance ratio: 252/1312 = 19.2%. <a href="https://github.com/ml-wsan/Meta-Adaptation" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)

    with col3:
        st.write('<p><strong>[J]</strong> Xia Cheng and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn2022_2.pdf" target="_blank"> Autonomous Traffic-Aware Scheduling for Industrial Wireless Sensor-Actuator Networks</a>, ACM Transactions on Sensor Networks, Vol. 19, Issue 2, pp. 38:1-38:25, February 2023. <a href="https://github.com/iiot-research/Autonomous-Scheduling" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Qi Li, Keyang Yu, Dong Chen, Mo Sha, and Long Cheng, <a href="https://users.cs.fiu.edu/~msha/publications/cns22.pdf" target="_blank"> TrafficSpy: Disaggregating VPN-encrypted IoT Network Traffic for User Privacy Inference</a>, IEEE Conference on Communications and Network Security (CNS), October 2022, acceptance ratio: 43/122 = 35.2%. </p>', unsafe_allow_html=True)
        st.write('<p><strong>[C]</strong> Junyang Shi and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icccn22.pdf" target="_blank"> Localizing Campus Shuttles from One Single Base Station Using LoRa Link Characteristics</a>, IEEE International Conference on Computer Communications and Networks (ICCCN), July 2022, acceptance ratio: 39/130=30.0%.<a href="https://github.com/junyang28/lorashuttlebus" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Junyang Shi, Xingjian Chen, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn2022.pdf" target="_blank"> Enabling Cross-technology Communication from LoRa to ZigBee in the 2.4 GHz Band</a>, ACM Transactions on Sensor Networks, Vol. 18, Issue 2, pp. 21:1-21:23, May 2022. <a href="https://github.com/junyang28/paper-ctclora" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Junyang Shi, Di Mu, and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/tosn21.pdf" target="_blank"> Enabling Cross-technology Communication from LoRa to ZigBee via Payload Encoding in Sub-1 GHz Bands</a>, ACM Transactions on Sensor Networks, Vol. 18, Issue 1, pp. 6:1-6:26, February 2022. <a href="https://github.com/junyang28/paper-lorabee" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Xiao Cheng and Mo Sha, <a href="https://users.cs.fiu.edu/~msha/publications/icnp21.pdf" target="_blank">  ATRIA: Autonomous Traffic-Aware Transmission Scheduling for Industrial Wireless Sensor-Actuator Networks</a>, IEEE International Conference on Network Protocols (ICNP), November 2021, acceptance ratio: 38/154 = 24.6%. <a href="https://github.com/iiot-research/Autonomous-Scheduling" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)
        st.write('<p><strong>[J]</strong> Junyang Shi, Mo Sha, and Xi Peng, <a href="https://users.cs.fiu.edu/~msha/publications/nsdi21.pdf" target="_blank"> Adapting Wireless Mesh Network Configuration from Simulation to Reality via Deep Learning based Domain Adaptation</a>, USENIX Symposium on Networked Systems Design and Implementation (NSDI), April 2021, acceptance ratio (fall deadline): 40/255 = 15.6%. <a href="https://github.com/aitianma/WSNConfDomainAdaptation" target="_blank">[source code and data]</a></p>', unsafe_allow_html=True)

    st.subheader("Project Links and Acknowledgments")

    st.write("""
    - **[Project Repository on GitHub](https://github.com/NotAPlumber505/WSN-Parameter-Project)**  
      Explore the source code, data, and detailed documentation for the project. Contribute to the development and improvement of the platform!

    - **[Project Website](https://users.cs.fiu.edu/~msha/CAREERproject.htm)**  
      Visit the official project website to learn more about the platform and explore its features.

    - **Funded by**  
      This research is funded by the National Science Foundation (NSF) and Florida International University (FIU).
    """)

    col4, col5 = st.columns(2)
    # Display Logos
    with col4:
        st.image("NSF_logo.png", width=150)
    with col5:
        st.image("fiu_logo.png", width=150)

    st.write("""
        The National Science Foundation (NSF) and Florida International University (FIU) represent the ongoing support that makes this groundbreaking research possible. 
        We acknowledge and thank both NSF and FIU for their generous contributions to advancing the field of industrial wireless sensor-actuator networks (WSANs). 
        Their collaboration has been instrumental in fostering innovation and enabling the development of this platform.
    """)

# Gap Analysis Tab
with tab2:
    st.header("Gap Analysis")

    # Sample data (replace with actual data from experiment)
    sim_val_acc = ['Training: Simulation Data\nTesting: Physical Data', 'Training: Physical Data\nTesting: Physical Data']
    sim_test_acc = 0.85  # Example value (replace with actual)
    phy_test_acc = 0.92  # Example value (replace with actual)

    # Accuracy values for each dataset
    phy_val_acc = [sim_test_acc, phy_test_acc]

    # Create dataframe
    gap_df = pd.DataFrame({
        'Dataset': sim_val_acc,
        'Accuracy': phy_val_acc
    })

    # User controls which visualizations to show
    st.subheader("Select Visualizations to Display:")
    single_source_selected_visuals = st.multiselect(
        "Choose one or more visualizations",
        ["Bar Chart", "Line Chart", "Scatter Plot", "Data Table"]
    )

    if single_source_selected_visuals:
        if "Bar Chart" in single_source_selected_visuals:
            fig = px.bar(gap_df, x='Dataset', y='Accuracy',
                         title="Simulation-to-Reality Gap",
                         labels={'Accuracy': 'Accuracy'},
                         color='Accuracy',
                         color_continuous_scale='Blues')
            st.plotly_chart(fig)

        if "Line Chart" in single_source_selected_visuals:
            line_data = generate_line_data()  #
            fig = px.line(line_data, x='Epoch', y='Accuracy',
                          title="Accuracy Progression",
                          labels={'Accuracy': 'Accuracy'})
            st.plotly_chart(fig)

        if "Scatter Plot" in single_source_selected_visuals:
            scatter_data = generate_scatter_data()
            fig = px.scatter(scatter_data, x='Simulated Prediction', y='Real Prediction',
                             title="Simulated vs Real Predictions",
                             trendline="ols",
                             labels={'Simulated Prediction': 'Simulated Accuracy', 'Real Prediction': 'Real Accuracy'})
            st.plotly_chart(fig)

        if "Data Table" in single_source_selected_visuals:
            st.subheader("Gap Analysis Table")
            st.dataframe(gap_df)

    else:
        st.info("No data selected to display.")

# Single Source vs Multi Source Tab
with tab3:
    st.header("Experiment Results: Single vs. Multi Source")

    # Load precomputed data
    data = load_experiment_data()

    if data is not None and not data.empty:
        # Visualization options (using the global chart type by default)
        st.subheader("Select Visualizations to Display:")
        single_source_selected_visuals = st.multiselect(
            "Choose visualizations to display for Single Source Results",
            [
                "Single Source Bar Chart", "Single Source Line Chart", "Single Source Table",
            ]
        )

        if single_source_selected_visuals:
            # 🔵 SINGLE SOURCE
            if "Single Source Bar Chart" in single_source_selected_visuals:
                fig = px.bar(data, x='Shots', y='Single Source Accuracy',
                             title="Single Source Accuracy (Bar)",
                             labels={'Single Source Accuracy': 'Accuracy'})
                st.plotly_chart(fig)

            if "Single Source Line Chart" in single_source_selected_visuals:
                fig = px.line(data, x='Shots', y='Single Source Accuracy',
                              title="Single Source Accuracy (Line)",
                              labels={'Single Source Accuracy': 'Accuracy'},
                              markers=True)
                st.plotly_chart(fig)

            if "Single Source Table" in single_source_selected_visuals:
                st.subheader("Single Source Table")
                st.dataframe(data[['Shots', 'Single Source Accuracy']])

        else:
            st.info("No data selected to display in Single Source Results.")

        multi_source_selected_visuals = st.multiselect(
            "Choose visualizations to display for Multi Source Results",
            [
                "Multi Source Bar Chart", "Multi Source Line Chart", "Multi Source Table",
            ]
        )

        if multi_source_selected_visuals:
            # 🔴 MULTI SOURCE
            if "Multi Source Bar Chart" in multi_source_selected_visuals:
                fig = px.bar(data, x='Shots', y='Multi Source Accuracy',
                             title="Multi Source Accuracy (Bar)",
                             color_discrete_sequence=['red'],
                             labels={'Multi Source Accuracy': 'Accuracy'})
                st.plotly_chart(fig)

            if "Multi Source Line Chart" in multi_source_selected_visuals:
                fig = px.line(data, x='Shots', y='Multi Source Accuracy',
                              title="Multi Source Accuracy (Line)",
                              color_discrete_sequence=['red'],
                              labels={'Multi Source Accuracy': 'Accuracy'},
                              markers=True)
                st.plotly_chart(fig)

            if "Multi Source Table" in multi_source_selected_visuals:
                st.subheader("Multi Source Table")
                st.dataframe(data[['Shots', 'Multi Source Accuracy']])

        else:
            st.info("No data selected to display in Multi Source Results.")


        combined_source_selected_visuals = st.multiselect(
            "Choose visualizations to display for Combined Source Results",
            [
                "Combined Bar Chart", "Combined Line Chart", "Combined Table",
            ],
        )

        # Data prep for combined view
        data_melted = data.melt(id_vars=['Shots'],
                                var_name='Source Type',
                                value_name='Accuracy')

        if combined_source_selected_visuals:
            # 🟣 COMBINED
            if "Combined Bar Chart" in combined_source_selected_visuals:
                fig = px.bar(data_melted, x='Shots', y='Accuracy',
                             color='Source Type', barmode='group',
                             title="Single vs. Multi Source Accuracy (Bar)",
                             labels={'Shots': 'Number of Shots', 'Accuracy': 'Accuracy Score'})
                st.plotly_chart(fig)

            if "Combined Line Chart" in combined_source_selected_visuals:
                fig = px.line(data_melted, x='Shots', y='Accuracy',
                              color='Source Type', markers=True,
                              title="Single vs. Multi Source Accuracy (Line)",
                              labels={'Shots': 'Number of Shots', 'Accuracy': 'Accuracy Score'})
                st.plotly_chart(fig)

            if "Combined Table" in combined_source_selected_visuals:
                st.subheader("Combined Source Table")
                st.dataframe(data_melted)

        else:
            st.info("No data selected to display in Combined Source Results.")
    else:
        st.error("No valid data available to display.")

# Meta Learning Tab
with tab4:
    st.header("Meta Learning Results")

    # First set of visualization options
    meta_learning_selected_visuals = st.multiselect(
        "Choose visualizations to display for Meta Learning Results",
        ["Bar Chart", "Line Chart", "Data Table"]
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

    # Second set of visualization options (Domain Adaptation Results)
    domain_adaptation_results_tab4 = st.multiselect(
        "Choose visualizations to display for Domain Adaptation Results",
        ["Bar Chart", "Line Chart", "Data Table"]
    )

    if domain_adaptation_results_tab4:
        # Bar Chart
        if "Bar Chart" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Bar Chart")

        # Line Chart
        if "Line Chart" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Line Chart")

        # Data Table
        if "Data Table" in domain_adaptation_results_tab4:
            load_accuracy_over_time_tab(chart_type="Table")
    else:
        st.info("No data selected to display Domain Adaptation Results.")