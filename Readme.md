# Introduction
This is the master repository for transportation modelling projects of [MMO lab](https://labmmo.ru/) 
### It is based on
- https://github.com/MeruzaKub/TransportNet
- https://github.com/tamamolis/TransportNet
- https://github.com/Lareton/transport_network_optimization

and also uses code written in other related projects of MMO lab. 

### Content
Repo contains implementations of basic algorithms for the equilibrium traffic assignment problem:
$$\sum_e \sigma_e(f_e) \to \min_{f \in F_d},$$

and the combined travel-demand (twostage) problem:
$$\gamma \sum_e\sigma_e(f_e) + \sum_{ij}d_{ij}\ln d_{ij} \to \min_{\substack{f\in F_d \\\ \sum_j d_{ij}=l_i\\ \sum_i d_{ij}=w_j}}.$$
# Installation
1. Grab bstabler's TransportationNetworks sumbodule: use `git clone  --recurse submodules`
or do `git submodule update --init` after clone
2. [Install](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) conda if not yet
3. Create and activate conda environment
```bash
conda env create -f environment.yml
conda activate tm
```
4. Add this conda environment to your jupyter notebook 
```bash
ipython kernel install --user --name=tm
```
After that you can select `tm` kernel from notebook's kernel menu. 
Alternatively, you can install  jupyter into the environment and run it from there (but it gave me an error while launching the notebook app)
```bash
conda install jupyter -n tm
```
More details about jupyter with conda env [here](https://stackoverflow.com/a/58068850)

Docker image  might be created on demand to simplify the installation process. We also have remote linux servers for internal use

# Experiments :

1. Пример запуска экспериментов : python3 compare_methods.py.
2. Модуль для запуска экспериментов src/test.py.
3. Сохранение результатов экспериментов в директорию experiments_result происходит, если выставить флаг save=True в методе test.plot().
4. В TransportationNetworks лежат датасеты городов. (загруженные из репозитория bstabler)
5. Алгоритмы расположены в my_algs.py и algs.py.
6. NFW вмержен в основной репозиторий mmo_tm.