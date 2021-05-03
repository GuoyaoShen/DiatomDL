# DiatomDL
A deep neural network structure to accelerate the design of diatom-inspired hierarchical arrangement of nanoparticle patterns.

### The schematic of the deep-learning model
<p align="center">
  <figure>
    <img src="https://github.com/GuoyaoShen/DiatomDL/blob/main/figs/whole_structure.png" width="400" />
    <figcaption>he schematic of the deep-learning model. The forward problem consists of a series of simulation data and a forward neural network. The simulation data includes designed structural parameters and their corresponding optical spectra. The neural network is trained by inputting the structural parameters and outputting the optical responses. After the training, the network is then sent to further solve the inverse problem. In this part, a large quantity of pseudo data is acquired by the neural network. With the targeted responses, the search can be performed inside the pseudo data to find the best match and give out the corresponding structural parameters./figcaption>
  </figure>
</p>

### Forward network structure
<p align="center">
  <img src="https://github.com/GuoyaoShen/DiatomDL/blob/main/figs/forward_structure.png" width="600" />
</p>

### Spectra search with pseudo dataset
<p align="center">
  <img src="https://github.com/GuoyaoShen/DiatomDL/blob/main/figs/spectra_search_structure.png" width="600" />
</p>
