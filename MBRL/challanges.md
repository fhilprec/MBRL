worldmodelPong.py pretty much all architectures working also with single step error of almost 0 but still compounding errors.
python MBRL/worldmodelPong.py PongLSTM render or python MBRL/worldmodelPong.py PongDreamer render
python MBRL/worldmodelDreamer.py 
python MBRL/worldmodelTransformer.py Transformer render



reward hardcoden
smaller rollout length 10
actor critic from dreamer paper https://arxiv.org/pdf/2010.02193





Pong mit perfect Policy trainiert -> worldmodel lernt den Ball mit der Kelle zu tracken anstatt den Actions zu folgen