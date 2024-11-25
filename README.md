
<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Problem Description
</h3>


**Context**

In healthcare systems worldwide, patient no-shows for scheduled medical appointments pose significant challenges. In Brazil, this issue is especially prevalent, with certain regions experiencing no-show rates exceeding 20%. These missed appointments result in inefficient utilization of medical resources, increased waiting times, and diminished healthcare accessibility for others. For hospitals and clinics, no-shows mean wasted time slots, underutilized staff, and missed opportunities to address patient needs.

**The Dataset**

The data used for this project comes from the public healthcare appointments system in Brazil and contains detailed records of **110,527 medical appointments**. Each record provides various attributes about the appointment, including:

- **Patient Demographics**: Age, gender, and neighborhood.
- **Health Factors**: Conditions such as hypertension, diabetes, alcoholism, and disabilities.
- **Appointment Scheduling Details**: Scheduled day, appointment day, and whether the patient received a reminder SMS.
- **Derived Features**: Additional insights such as the day of the week of the appointment and lead time (days between scheduling and appointment) have been carefully engineered to enhance predictive power.
- **Appointment Outcomes**: Whether the patient showed up or not.
- **Source:** The dataset can be found along with its data dictionary at <https://www.kaggle.com/datasets/joniarroba/noshowappointments>. I have included the dataset in the project directory.

**The Problem**

Missed appointments strain the healthcare infrastructure and create inequities in patient care delivery. Predicting which patients are likely to miss their appointments can enable hospitals to:

1. **Optimize Resources**: Allocate medical personnel and equipment more efficiently.
2. **Intervene Early**: Send reminders or personalized follow-ups to patients likely to miss their appointments.
3. **Improve Accessibility**: Open unused slots to other patients, reducing waiting times.

The no-show behavior of patients is shaped by diverse factors, including their health conditions, previous attendance patterns, appointment scheduling habits, and external influences like reminders. These factors reflect not just individual tendencies but also systemic issues like accessibility and scheduling practices. Addressing this problem requires careful consideration to avoid introducing biases based on patient demographics or socioeconomic status.

**Solution:Project Objective**

This project seeks to develop a predictive solution that can determine the probability of a no-show for each scheduled appointment. The goal is to create a **scalable, interpretable, and actionable machine learning model** that:

1. **Accurately Predicts No-Shows**: The model leverages historical patterns, such as a patient's past attendance behavior (e.g., previous and missed appointments), along with other critical factors, such as medical history, appointment timing, and external influences like SMS reminders.
2. **Enhances Decision-Making**: By providing precise probabilities and actionable insights, healthcare providers can focus resources on patients at higher risk of no-showing.
3. **Patient Welfare**: By having a predictive system of No-shows at appointment day in place, an efficient mechanism for ensuring patient attendance by intimation, follow-ups and other effective measures. As a consequence, there would be a uplift in the patient welfare of the area concerned by a having good health issue addressal of the population.

<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Exploratory Data Analysis
</h3>


**Cleaning and preparation** of the dataset have been done as prerequisites to the EDA here involving:

- Loading and checking basic information of the dataset
- Checking null and duplicates in the dataset
- Sanitizing column names
- Assigning appropriate datatypes to featrures/columns
- Target feature encoding

For the **Exploratory Data Analysis** , starting with drawing inference from basic statistical data, I move on to follow very closely with our course tutorials for an extensive EDA involving:

- Target feature analysis
- Analyze feature importance of the categorical features with Risk ratio
- Analyze feature importance of the categorical features with Mutual Information Score
- Feature importance analysis of the continuous features through ROC AUC score
- Feature importance analysis of the continuous features through Correlation
- Removing anomalous condition data and outliers of features.
- Feature Engineering with dates to get patient history and waiting time (for appointment), feature combination for handicap and cumulative health issues

_These exercises could be found in my_ **_notebook.ipynb_**


<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Model Training
</h3>


As a pre-requisite of the model training, I had to:

- Encoding of the gender feature as it was not viable at the earlier EDA time as it would have affected the feature importance evaluation.
- Dropping of features deemed not required for training the model.
- Splitting of the dataset into full-train, test, train and validation with target features split accordingly.

Then the model training ensued for training multiple models and tuning their parameters following closely the approach taught in our course tutorials which include the following:

- Evaluating best parameters for the 4 classifiers (Logistic Regression, Decision Tree, Random Forest & XGBoost) with **train** and **validation** dataset.
- With best evaluated parameters for each of the 4 classifier, trained each classifier model with 5 fold cross validation on the **full- train** dataset.
- Although, for the XGBoost model cross validation was not used.
- Finally, with hyper-tuned parameters evaluated from the above steps, all the 4 classifiers were trained on the **unseen test data.** Based on the metrics of this training, the final model (XGBoost) was selected.

_These exercises could be found in my_ **_notebook.ipynb_**



<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Exporting notebook to script
</h3>


All data structure and data transformation including their preparation & cleaning, and feature engineering, model training of the best evaluated model with hyper-tuned parameters have been exported from the notebook in the form of a script namely **train.py** Running this script will outcome:

- Generation of the **model_final.bin** file which holds the final trained model
- Generation of the file **cleaned_prepared_df.pkl** which shall be used by the test script for providing feature engineered patient historical data for transformation of raw test input data sample for the model.



<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Dependency and environment management
</h3>


_All project dependencies are listed in the_ **_Pipfile_**

1\. Go the your wsl environment from you powershell terminal with administrator privilege. You should land in your WSL home directory by default.

    C:\\> _wsl

2\. If you do not have pipenv installed already, you can install it by with the command

    pip install pipenv

3\. From your home directory at WSL, clone my GitHub project repository with the link I submitted

    git clone &lt;link_to_my_project_directory_at_Github&gt;.git

4\. Go inside that cloned directory

    cd midtermproject-2024-mlz

5\. Install all dependencies/packages mentioned in the **Pipfile** within the new virtual environment being created (as pipenv will prioritize Pipfile, requirements.txt has not been used)

    pipenv install

6\. Activate the new virtual environment created:

    pipenv shell

7\. In your virtual environment, form within the project directory, run the jupyter notebook

    jupyter notebook

8\. You shall get some urls like the following in the response (please note these URL are only indicative of the pattern and will be different in your case)

\[I 2024-11-24 13:56:40.430 ServerApp\] <http://localhost:8888/tree?token=a50930ee6fe34cbf934d8ba0460bdf942edc8f21d87df4a0>

\[I 2024-11-24 13:56:40.430 ServerApp\] <http://127.0.0.1:8888/tree?token=a50930ee6fe34cbf934d8ba0460bdf942edc8f21d87df4a0>


<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Reproducibility
</h3>


1. From within the virtual environment from inside the project directory, kindly run the **train.py** script to train the best evaluated model on the dataset from **KaggleV2-May-2016.csv** and save the model
    *python train.py*


2. Use one of the links (you got when started the Jupyter notebook) in the browser to open the Jupyter notebook . From the Jupyter notebook GUI, you can upload my **notebook.ipynb** in the project directory and review it.


<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Model Deployment
</h3>


1. Please run the **gunicorn** WSGI http server to make the flask web-service (**predict.py)** serving the model, active for consumption/use by running this command inside the virtual environment from inside the project folder

        gunicorn --bind 0.0.0.0:9696 predict:app

2. When the gunicorn server had started successfully, open another powershell window, go to WSL and cd to project folder. Then activate the virtual environment. Now, you can run the **test.py** from within the virtual environment from inside the project directory to see if the deployed model is served through the web service and what the model predicts.

        python test.py



<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Containerization
</h3>

1. Install Docker Deskop on you Windows 11 (currently I have Docker Desktop version 4.35.0). If you have other set up, you may install docker accordingly (Mac/Linux/Windows 10)

2. In the settings of Docker Desktop, in the 'General' tab/menu please ensure you have "Use the WSL 2 based engine (Windows Home can only run the WSL 2 backend)" checked/ticked.

3. Again, in the settings, "WSL integration" sub menu in the "Resources" menu/tab, please ensure "Enable integration with my default WSL distro" is checked/ticked. Further ensure that the "Enable integration with additional distros:" slider button is turned on.

4. Start the Docker Engine in the Docker Desktop, if not already started.

5. Now from the project directory in the WSL (this should not be from the pipenv virtual environment; so don't activate it this time), issue the command to build the docker image. The image would be built as mentioned in the submitted **Dockerfile**.

        docker build -t midterm-2024 .

6. After the image is built and the application successfully containerized, we can list the image from the WSL by following command

        docker images

7. Now run the containerized application from the project folder (outside virtual environment)

        docker run -it --rm -p 9696:9696 midterm-2024

8. Now activate virtual environment in another WSL tab from inside the project directory and run the **test.py** from that virtual environment from inside the project directory to get the **predict service** from the containerized application

       python test.py



<h3 style="background-color:#f0f0f0;color:#333;padding:10px;border-radius:5px;text-align:center;">
Cloud Deployment
</h3>


1\. Navigate to project directory and issue the following command to install awsebcli (Amazon Web Service Elastic Beanstalk Command Line Interface)

    pipenv install aswebcli --dev

2\. Activate your virtual environment from the project directory and issue following command to create your AWS EB application from your virtual environment form inside the project directory (my application name is midterm-2024-cloud and region is eu-north-1)

    eb init -p docker -r eu-north-1 midterm-2024-cloud

3\. After you application is successfully created, from the virtual environment from within the project directory, issue following command to launch the environment. On successful launching of the cloud environment, We shall also get the URL/Domain name here starting with "Application available at &lt;generated_URL&gt;

    eb create midterm-2024-cloud --enable-spot

4\. I have used this URL in a test script namely cloud_test.py to test the containerized application hosted at the AWS EB. We run it to check if our dockerized application in the cloud is working is properly and the model is serving the prediction.

    python cloud_test.py



**_Note_**_: I have taken required screenshot like testing/interaction with local, containerized local, cloud deployed services, deployed service running at AWS EB, etc. They can be found at the screenshot directory inside the project directory_