* technically I'm constructing some new features on top of ‘sex’, ‘age’,‘sibsp’, ‘parch’ and ‘embarked’ and train model with them. I choose this approach because it was interesting
* I use fuzzy match with sorting the tockens. It gives me a result in any case which probably good in a case of "muted" requirements


to install and run the app you need
```
make deps
make server
```
also I expect that you have python2 and pip installed on your machine

There are some warnings, I decided no to spend time on fixing them

For testing I prefer to use curl:
```
> curl http://127.0.0.1:5000/survive\?name\=Emily%20R
{"name":"Rugg, Emily","probability":0.11002975455916628}
```
