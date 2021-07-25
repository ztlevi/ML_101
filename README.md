# Introduction

Github repo: [https://github.com/ztlevi/ML\_101](https://github.com/ztlevi/ML_101)

My [Gitbook workspace](https://app.gitbook.com/@ztlevi).

Gitbook is deployed over [here](https://ztlevi.gitbook.io/ml-101). Old link is [here](https://git.io/fj0yP).

All codes are available over [here](https://github.com/ztlevi/Machine_Learning_Questions/tree/master/codes).

## Start to contribute

```bash
git clone https://github.com/ztlevi/ML_101.git

cd ML_101

# Pre-commit plugins
pip3 install pre-commit
pre-commit install

# Please use nodejs 10
brew install nodenv node-build
nodenv install 10.23.2
nodenv local 10.23.2 # set local nodejs version

# Install dependencies
npm install
npm run docs:prepare

# Start to watch the book
npm start # or npm run docs:watch

# Deploy the book
npm run docs:publish
```

## Use the code

```bash
pip3 install pipenv
pipenv install
pipenv shell
```

