
from spellchecker import SpellChecker
spell = SpellChecker(language='de')
string = "blinken nachricht sichtbar ref 0zid blinken empfangslampe tc0 einschalten ton persönlich karte nachsehen aktuell nachrichten manager herrn steinegger passieren erhalten nachrichten jahren tc0 melden gelb karte richtig situation anschauen voraus"

for l in string.split(' '):
	print(spell.correction(l))