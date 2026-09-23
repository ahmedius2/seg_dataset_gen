# How to use

1. Generate the blender scenes by running the command below. Note that you will need the source\_scene.blend file since scene are generated using it. Before running, check dataset\_config.py
and make any modifications you see necessary such as adjusting the paths.
```
cd lidar
blenderproc run lidar_generate_dataset.py
cd ..
```
This will create .scene files and place them in a folder named generated\_scenes.

2. Next is to export these scenes into model files which gazebo can use. Adjust the path in the script to point the correct directory before running.
```
cd gazebo
./export_all.sh
```
Now you have custom\_models and custom\_worlds directories which gazebo can import. Update GZ\_SIM\_RESOURCE\_PATH environment variable so gazebo have access to these directories.

3. Run the below command to create the dataset by running simulations on gazebo:
```
./runsimall.sh
cd ..
```
This will create the simulation\_data and dnn\_dataset folders.

4. Now you can train the dnn that does ground segmentation. 
```
cd lidar
python -m train_dnn.train
```
