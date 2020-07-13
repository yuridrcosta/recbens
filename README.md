# recbens
### Teste disponível em: https://recbens.herokuapp.com/
recbens é um módulo python para reconhecimento de bens em textos na língua portuguesa. Baseado no conceito de Reconhecimento de Entidades Nomeadas (<i>Named Entity Recognition</i>), esse módulo fornece ao usuário um modelo com acurácia de <b>96,75%</b> em textos da língua portuguesa, incluindo todas as ferramentas necessárias para facilitar a correção e melhoria do próprio modelo.<br/> A documentação completa pode ser encontrada em: https://recbens.herokuapp.com/docs/<br/><br/>
- Exemplo de uso:
```
import recbens

with open('meu_texto.txt') as input_file:
  text = input_file.read()

model = recbens.Model()
dictionary = model.load_dictionary()
evaluation = model.evaluate(model,text,dictionary)

# Isso vai lhe retornar uma lista de tuplas com a classificação para cada palavra entre bem ou não
```
