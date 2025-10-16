# How we run the Visual Narrator
Since the Dockerfile stated that the python version is python3.6 since this is not a version that is easy downloadable we opted for running it in a docker container. 

## Steps
Make sure you have docker running (e.g. with Docker Desktop)
### Build the Dockerimage 
```
docker build -f Dockerfile -t acedesign/visualnarrator .
```

### Run the input
Within the VisualNarrator folder we created an input folder and put all the input files in there. In that way we could loop over the folder and run all the files in one go.

```
$image = "acedesign/visualnarrator:latest"
>> Get-ChildItem -Path .\input\*.txt | ForEach-Object {
>>   $rel = $_.FullName.Substring((Get-Location).Path.Length).TrimStart('\')
>>   $containerPath = "/usr/src/app/output/$($rel -replace '\\','/')"
>>   docker run --rm --mount type=bind,source="${PWD}",target=/usr/src/app/output $image $containerPath     
>> }
```

Using this the folder System will be created in the same directory as the input folder. Within that folder two folders were created "ontology" with .omn files and "reports" with .html files. 

Adding the -t flag to the docker run command it is possible to change the threshold which is defaulted to 1.

Example with threshold set to 2
```
$image = "acedesign/visualnarrator:latest"
>> Get-ChildItem -Path .\input\*.txt | ForEach-Object {
>>   $rel = $_.FullName.Substring((Get-Location).Path.Length).TrimStart('\')
>>   $containerPath = "/usr/src/app/output/$($rel -replace '\\','/')"
>>   docker run --rm --mount type=bind,source="${PWD}",target=/usr/src/app/output $image $containerPath -t 2  
>> }
```

### Extract classes
We added our own script to extract the classes from the ontology files. It is located in the same directory as the System and input folders with the name "create_outputfiles.py". To run it:
```
py create_outputfiles.py
```

## Footnote
The way the folders and files are structured here are thus not the same as while running the experiment.