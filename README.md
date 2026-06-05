# template-mdf

Repositório com notebooks de exemplos para estudos computacionais em métodos
numéricos, com ambiente reprodutível usando
[Pixi](https://pixi.prefix.dev/latest/), Jupyter e Jupytext.

O foco atual está em problemas de ODE e PDE resolvidos com ferramentas do
ecossistema científico Python, em particular `scipy` e `py-pde`.

## Objetivos

- Organizar notebooks didáticos de métodos numéricos.
- Manter exemplos executáveis do começo ao fim.
- Facilitar reprodução de resultados com um ambiente Pixi versionado.
- Sincronizar notebooks `.ipynb` com arquivos `.py` em formato Jupytext.
- Comparar métodos numéricos por erro, convergência, estabilidade e custo.

## Estrutura do repositório

- `pixi.toml`: define dependências, ambientes e tarefas do projeto.
- `pixi.lock`: registra as versões resolvidas das dependências.
- `jupytext.toml`: configura o pareamento entre `.ipynb` e `.py:percent`.
- `scripts/sync_notebooks.py`: sincroniza notebooks usando Jupytext.
- `notebooks/examples/ode/`: exemplos de ODE.
- `notebooks/examples/pde/`: exemplos de PDE.

## Notebooks disponíveis

### ODE

- `notebooks/examples/ode/01-intro-scipy-ode.ipynb`: introdução à solução de
  ODEs com `scipy`, métodos de passo fixo, convergência e Lotka-Volterra.
- `notebooks/examples/ode/02-problemas-rigidos.ipynb`: sistemas rígidos,
  Lorenz, Euler explícito/implícito, métodos adaptativos e comportamento em
  tempos longos.
- `notebooks/examples/ode/03-ciclos-estabilidade-bifurcacoes.ipynb`: ciclos,
  estabilidade, bifurcações e transições qualitativas em sistemas
  predador-presa.

### PDE

- `notebooks/examples/pde/01-intro-py-pde.ipynb`: introdução à `py-pde` com
  Poisson 1D estacionário/transiente, análise de convergência, comparação de
  solvers e um bônus com Lotka-Volterra.
- `notebooks/examples/pde/02-poisson-2d-py-pde.ipynb`: extensão para Poisson 2D
  estacionário/transiente, visualização de campos, cortes e mapas espaço-tempo.

## Ambientes Pixi

O projeto define dois ambientes principais:

- `default`: ambiente base para notebooks com `numpy`, `scipy`, `pandas`,
  `matplotlib`, `jupyterlab`, `jupytext` e ferramentas de formatação.
- `pypde-env`: ambiente com as dependências do `default` mais a biblioteca
  `py-pde`.

Para os notebooks de ODE, o ambiente `default` é suficiente. Para os notebooks
de PDE, use o ambiente `pypde-env`.

## Como usar

### 1. Instale o Pixi

Siga a documentação oficial:

<https://pixi.prefix.dev/latest/installation/>

Depois da instalação, confira se o Pixi está disponível:

```sh
pixi --version
```

### 2. Clone o repositório

```sh
git clone git@github.com:volpatto/template-mdf.git
cd template-mdf
```

### 3. Instale as dependências

```sh
pixi install --frozen
```

### 4. Abra os notebooks de ODE

```sh
pixi shell
jupyter lab
```

### 5. Abra os notebooks de PDE

```sh
pixi shell -e pypde-env
jupyter lab
```

No VS Code, selecione o kernel Python correspondente ao ambiente Pixi usado.

## Fluxo de trabalho recomendado

Antes de editar notebooks, atualize o repositório:

```sh
git pull
```

Depois, ative o ambiente adequado:

```sh
pixi shell
```

ou, para notebooks com `py-pde`:

```sh
pixi shell -e pypde-env
```

Ao terminar alterações, sincronize os notebooks e rode as verificações:

```sh
pixi run precommit-sync
```

## Tarefas Pixi úteis

Lista os notebooks encontrados em `notebooks/`:

```sh
pixi run notebooks-smoke
```

Sincroniza notebooks `.ipynb` com Jupytext:

```sh
pixi run notebooks-sync
```

Executa as verificações do `pre-commit`:

```sh
pixi run precommit
```

Sincroniza os notebooks e executa as verificações:

```sh
pixi run precommit-sync
```

## Jupytext

Os notebooks são mantidos em pares:

- `.ipynb`: arquivo aberto no Jupyter/VS Code.
- `.py`: representação textual em formato percent, mais fácil de revisar em
  diffs.

A configuração está em `jupytext.toml`:

```toml
formats = "ipynb,py:percent"
```

Para sincronizar manualmente:

```sh
pixi run notebooks-sync
```

## Arquivos gerados localmente

O `.gitignore` ignora arquivos comuns de saída, como:

- `tmp/`
- `outputs/`
- `refs/`
- arquivos `.csv`
- figuras `.png` dentro de `notebooks/`

Se algum resultado for essencial para uma aula, prefira documentar no notebook
como reproduzi-lo.

## Problemas comuns

### O comando `pixi` não foi encontrado

Feche e abra novamente o terminal. Se o problema continuar, revise a instalação
do Pixi.

### O notebook não encontra uma biblioteca Python

Confira se o ambiente correto está ativo. Para notebooks com `py-pde`, use:

```sh
pixi shell -e pypde-env
```

### As verificações falharam antes do commit

Rode:

```sh
pixi run precommit-sync
```

Depois revise os arquivos modificados e execute novamente o comando se
necessário.

## Licença

Este projeto usa a licença MIT. Veja o arquivo `LICENSE`.

## Declaração de uso de IA

A revisão, refatoração e implementação deste repositório foi/é assistida por IA.
LLMs utilizadas:

- OpenAI Codex;
- Github Copilot.

## Apoio institucional

Este projeto recebe apoio institucional do
[Laboratório Nacional de Computação Científica (LNCC)](https://www.gov.br/lncc/pt-br).

<p align="left">
  <a href="https://www.gov.br/lncc/pt-br">
    <img src="resources/logo/lncc-mcti.svg" alt="Logo do LNCC" width="240">
  </a>
</p>
