"""
Author: Benjamin M. Sainz-Tinajero @ Tecnologico de Monterrey, 2022.

"""

from gen import *


ds = ['absenteeism-at-work', 'arrhythmia', 'breast-cancer-wisconsin', 'breast-tissue', 'car-evaluation', 'dermatology',
      'echocardiogram', 'ecoli', 'forest', 'forest-fires', 'german-credit', 'glass', 'hepatitis', 'image-segmentation',
      'ionosphere', 'iris', 'leaf', 'liver', 'parkinsons', 'seeds', 'segment', 'sonar', 'soybean-large',
      'student-performance', 'tic-tac-toe', 'transfusion', 'user-knowledge-modeling', 'wine', 'yeast', 'zoo']
    
      
nature = ['canada', 'coast', 'highway-in-the-desert', 'london', 'parking-lot', 'port-city', 'port', 'road-with-trees', 
          'varadero', 'white-containers']
        
if __name__ == "__main__":
    for d in ds:
        run_mocle(d, 'auto', runs=10, max_gens=50)
    
