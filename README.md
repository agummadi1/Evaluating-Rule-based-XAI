<h1>Evaluating Rule-based XAI Techniques</h1>

<h2>Rule-based XAI Techniques</h2>

<h3>(1) ANCHOR</h3>
Anchor Explainability is a rule-based method for interpreting machine learning models. Unlike traditional feature importance methods, Anchors provide if-then conditions that serve as high-precision explanations for model predictions. These conditions, called anchors, define specific feature values or ranges that, when met, result in a consistent prediction with high confidence.
Anchors work by iteratively perturbing input features and observing how the model's predictions change. The goal is to find the minimal set of conditions that "anchor" a prediction, meaning the model remains highly confident in its decision within this feature subset. 

<h3>(2) RuleFit</h3>
RuleFit is a hybrid interpretable model that combines linear regression with decision rules extracted from tree ensembles (like decision trees or random forests).It generates simple, human-readable if-then rules alongside linear terms, making it easy to understand how features contribute to predictions.

<h2>Datasets</h2>

<h3>(1)  MEMS datasets:</h3>
To build these datasets, an experiment was conducted in the motor testbed to collect machine condition data (i.e., acceleration) for different health conditions. During the experiment, the acceleration signals were collected from both piezoelectric and MEMS sensors at the same time with the sampling rate of 3.2 kHz and 10 Hz, respectively, for X, Y, and Z axes. Different levels of machine health condition can be induced by mounting a mass on the balancing disk, thus different levels of mechanical imbalance are used to trigger failures. Failure condition can be classified as one of three possible states - normal, near-failure, and failure. Multiple levels of the mechanical imbalance can be generated in the motor testbed (i.e., more masses indicate worse health condition). In this experiment, three levels of mechanical imbalance (i.e., normal, near-failure, failure) were considered Acceleration data were collected at the ten rotational speeds (100, 200, 300, 320, 340, 360, 380, 400, 500, and 600 RPM) for each condition, while the motor is running, 50 samples were collected at 10 s interval, for each of the ten rotational speeds. We use this same data for defect-type classification and learning transfer tasks.
<h3>(2) N-BaIoT dataset:</h3>
It was created to detect IoT botnet attacks and is a useful resource for researching cybersecurity issues in the context of the Internet of Things (IoT). This data was gathered from nine commercial IoT devices that were actually infected by two well-known botnets, Mirai and Gafgyt.
Every data instance in the dataset has access to a variety of features. These attributes are divided into multiple groups:

A. Stream Aggregation: These functions offer data that summarizes the traffic of the past few days. This group's categories comprise: H: Statistics providing an overview of the packet's host's (IP) recent traffic. HH: Statistics providing a summary of recent traffic from the host (IP) of the packet to the host of the packet's destination. HpHp: Statistics providing a summary of recent IP traffic from the packet's source host and port to its destination host and port. HH-jit: Statistics that summarize the jitter of the traffic traveling from the IP host of the packet to the host of its destination.

B. Time-frame (Lambda): This characteristic indicates how much of the stream's recent history is represented in the statistics. They bear the designations L1, L3, L5, and so forth.

C. Data Taken Out of the Packet Stream Statistics: Among these characteristics are:

Weight: The total number of objects noticed in recent history, or the weight of the stream.

Mean: The statistical mean is called the mean.

Std: The standard deviation in statistics.

Radius: The square root of the variations of the two streams.

Magnitude: The square root of the means of the two streams.

Cov: A covariance between two streams that is roughly estimated.

Pcc: A covariance between two streams that is approximated.

The dataset consists of the following 11 classes: benign traffic is defined as network activity that is benign and does not have malicious intent, and 10 of these classes represent different attack tactics employed by the Gafgyt and Mirai botnets to infect IoT devices.

benign: There are no indications of botnet activity in this class, which reflects typical, benign network traffic. It acts as the starting point for safe network operations.

gafgyt.combo: This class is equivalent to the "combo" assault of the Gafgyt botnet, which combines different attack techniques, like brute-force login attempts and vulnerability-exploiting, to compromise IoT devices.

gafgyt.junk: The "junk" attack from Gafgyt entails flooding a target device or network with too many garbage data packets, which can impair operations and even result in a denial of service.

gafgyt.scan: Gafgyt uses the "scan" attack to search for IoT devices that are susceptible to penetration. The botnet then enumerates and probes these devices in an effort to locate and compromise them.

gafgyt.tcp: This class embodies the TCP-based attack of the Gafgyt botnet, which targets devices using TCP-based exploits and attacks.

gafgyt.udp: The User Datagram Protocol (UDP) is used in Gafgyt's "udp" assault to initiate attacks, such as bombarding targets with UDP packets to stop them from operating.

mirai.ack: To take advantage of holes in Internet of Things devices and enlist them in the Mirai botnet, Mirai's "ack" attack uses the Acknowledgment (ACK) packet.

mirai.scan: By methodically scanning IP addresses and looking for vulnerabilities, Mirai's "scan" assault seeks to identify susceptible Internet of Things (IoT) devices.

mirai.syn: The Mirai "syn" attack leverages vulnerabilities in Internet of Things devices to add them to the Mirai botnet by using the SYN packet, which is a component of the TCP handshake procedure.

mirai.udp: Based on the UDP protocol, Mirai's "udp" attack includes bombarding targeted devices with UDP packets in an attempt to interfere with their ability to function.

mirai.udpplain: This class represents plain UDP assaults that aim to overload IoT devices with UDP traffic, causing service disruption. It is similar to the prior "udp" attack by Mirai.

The dataset consists of data collected from 9 IoT devices, however, for this paper, we have chosen to specially work on the dataset of DEVICE 7 - Samsung SNH 1011 N Webcam which has only classes 1 -6

<h2>Metrics</h2>

Descriptive Accuracy, Sparsity, Stability, Efficiency, Coverage, Precision and Fidelity

<h2>How to run the programs</h2>

Inside each rule-based method's folder, you will find dataset+metric based folders.

<h3>The 4metrics folder</h3>

It contains the codes of all 4 models used in the paper. Each one of these programs outputs:

a. The accuracy for the each model/method

b. The values for y_axis for the sparsity (To generate Sparsity Graphs, you will need to copy-paste these values into the respective gen_sparsity code).

c. Top features in importance order based on rules 
<\n>(To generate Descriptive Accuracy Graphs, rerun code using same number of samples but omitting top features each time and note the accuracy. Enter these accuracies in the respective gen_desc_acc code)
<\n>(For Stability metrics, run the programs 3x or more and note the obtained top k features in each run and compare the similarity.)

d. Time taken for execution (Efficiency)

e. The number of samples being used (Change value and rerun as desired, to obtain varied efficiency results)

<h3>The Fidelity folder</h3>

It contains the codes of all 4 models used in the paper. Each one of these programs outputs the Fidelity score. Anchor codes also have binary Precision & Coverage scores

<h3>The Multiclass folder</h3>
  
It contains the codes of all 4 models used in the paper. Each one of these programs outputs a class-wise output file which consists of:

a. Rules

b. Precision

c. Coverage
