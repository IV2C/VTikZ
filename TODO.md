# TODO

- [X] either make a script to add an entry in the dataset or just doucemnt the command to do it.
- [X] diff computation
- [X] Make the dataset with a python script => put it in a github workflow file
  - Create a new entry in the dataset:
    - Add submodule corresponding to the split(tikz,svg,etc) with the name "input" `git submodule add --name dog2 git@github.com:VarBench-SE/tikz.git dog2/input`
    - Copy the folder of the submodule to a directory with the name reference and delete the .git in that folder
  - publish the dataset
    - Compute the diffs with the script(diffs between input and reference, ignoring .git)
    - For each entry/repo
      - Commit and push the changes
      - find the commit pointed 
      - Add in the dictonnary : the name of the repo, the commit hash, the prompt, and the diff
    - Create a dataset from the dictionnary and push it to huggingface
- [ ] Make run_evaluation file
  - proper prompt template
- [ ] ascii art "compiler"
- [ ] github action
  - [ ] create a store docker images(for now one docker image is stored in docker.io)
  - [ ] creates and publishes the dataset
- dockerize execution


## Side notes
- maybe I will need to change the patches manually to replace the paths(for now there is a "renaming" of input by reference, I might need to replace all reference/... to input/ to prevent the renaming), but it might not be necessary
