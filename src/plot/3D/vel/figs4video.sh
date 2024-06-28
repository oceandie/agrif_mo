#!/bin/bash

fig1="_aviso-gosi10p0.png"
fig2="_agrif-gosi10p0.png"

for yyyy in {2005..2007}; do
    yy1="$yyyy"
    for mm in {1..12}; do
        if (( mm < 10 )); then
           mm1="0$mm"
        else
           mm1="$mm"
        fi
        date=${yy1}${mm1}
        echo $date

        # Add text
        convert -font Helvetica -pointsize 80 -draw "text 100,100 'Date: ${yy1}"-"${mm1}'" ${date}${fig1} t${date}${fig1}
        convert -trim t${date}${fig1} t${date}${fig1}
        convert -trim ${date}${fig2} t${date}${fig2}
        convert "t${date}${fig1}" "t${date}${fig2}" +append "${date}.png" 
        rm t${date}${fig1} t${date}${fig2}
    done
done
