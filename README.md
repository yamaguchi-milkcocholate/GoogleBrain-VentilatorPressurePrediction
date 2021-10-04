# GoogleBrain-VentilatorPressurePrediction

# install pyenv
Ubuntu
https://qiita.com/neruoneru/items/1107bcdca7fa43de673d
```
git clone https://github.com/pyenv/pyenv.git .pyenv

cd .pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> .bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> .bashrc
echo 'eval "$(pyenv init --path)"' >> .bashrc
source .bashrc

pyenv install 3.9.0

pyenv glogal 3.9.0
```

# install pipenv
```
pip install pipenv
cd GoogleBrain-VentilatorPressurePrediction
pipenv shell
```

# install kaggle
```
pipenv install kaggle

# download kaggle.json to ~/.kaggle
# https://www.kaggle.com/teppeiyamaguchi/account


```