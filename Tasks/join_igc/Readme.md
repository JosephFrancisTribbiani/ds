[Ссылка](https://xp-soaring.github.io/igc_file_format/igc_format_2008.html#link_4.1) на описание формата IGC

B record description:

**B H H M M S S D D M MM MM N D D D M MM MM E V P P P P P G G G G G CR LF**

Example:

B0836345440059N03758505EA0014800169

|B record Decription|Size|Element|Example|Remarks|
|:-:|:-:|:-:|:-:|:-:|
|Time UTC|6 bytes|HHMMSS|083634|Время UTC|
|Latitude|8 bytes|DDMMmmmN/S|5440059N|Широта|
|Longitude|9 bytes|DDDMMmmmE/W|03758505E|Долгота|
|Fix validity|1 byte|A or V|A||
|Pess Alt.|5 bytes|PPPPP|00148|Высота над уровнем моря (1013.25 HPa)|
|GNSS Alt.|5 bytes|GGGGG|00169||

`DD` (`DDD`) в формате Широты (Долготы) означает значение Degrees (градусы)  
`MM` (`MM`) в формате Широты (Долготы) означает значение Minutes (минуты)  
`mmm` (`mmm`) в формате Широты (Долготы) означает значение долей минут  
`N/S` (`E/W`) в формате Широты (Долготы) означает полушарие

Например: `5440059N` (формат `DDMMmmmN/S`) - северная широта `54` градуса, `40.059` минут

Перевести в формат `D.d` (Decimal Degrees): $.d =  M.m / 60$.  
Например, `5440059N` (формат `DDMMmmmN/S`) -> `54.66765N` (формат `D.dN/S`): $.d = 40.059 / 60 = 0.66765$

[Ссылка](https://www.directionsmag.com/site/latlong-converter/) на конвертер