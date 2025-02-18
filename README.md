![Framework_figure](https://github.com/user-attachments/assets/32d6899f-59e6-4d09-a2f3-aae68bbe9349)

The first step of the FedTPS framework involves training the diffusion-based generative model, whose code lies under the TFDiff directory. 

Once trained, it can be used to generate synthetic data to augment the traffic prediciton models, whose code can be found under the FedTPS framework. 


To train the TFDiff model, install the packages in the requirements.txt file. Then, navigate to the TFDiff file, and run the command

``
\\python main.py --num_clients 2 --city chengdu
``

The trained model can then be used to generate synthetic data in the format specified in FedTPS/gen_data dir. 

To train the traffic prediction models, navigate to the FedTPS directory, and run the command

``
python train_fl.py --num_clients 2 --model_name STGCN --epochs 80 --city chengdu --FL_method FedAvg --gen_samples 480
``

where you can replace the model_name and FL_method argument with another suitable model and method. 
