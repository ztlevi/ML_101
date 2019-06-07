# [Machine Learning Questions](https://git.io/fj0yP)

## Rendering Issue

Sometimes the mathjax tends to perform slowly, and it takes some time to fully render all math formula. In this case, you just need to refresh your browser.

## Start to contribute

```sh
git clone https://github.com/ztlevi/Machine_Learning_Questions.git

cd Machine_Learning_Questions

# Pre-commit plugins
pip install pre-commit
pre-commit install

# Install Nodejs first
# Install dependencies
npm install
npm run docs:prepare

# Start to watch the book
npm start # or npm run docs:watch

# Deploy the book
npm run docs:publish
```
