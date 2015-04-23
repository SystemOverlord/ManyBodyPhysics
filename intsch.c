void intsch(double (*fct)(double),double anf, double aend, double h, 
            double gen, int anzmax, double nullst[], int *anz)
/*	    
 Dieses Programm macht eine Grobsuche im Intervall [anf,aend] mit der
 Schrittweite h. Im Falle eines Vorzeichenwechsels der Funktion 'fct' 
 wird solange eine Intervallschachtelung durchgefuehrt, bis die Nullstelle
 mit der (i.a. relativen) Genauigkeit 'gen' bekannt ist.
 Alle Nullstellen werden im eindim. Feld 'nullst' gespeichert, und dieses
 Feld und die Zahl der gefundenen Nullstellen ('anz') werden an das
 aufrufende Programm zurueckgegeben.
 Im Falle eines Ueberlaufs des Feldes 'nullst', d.h. wenn 'anz>anzmax',
 wird eine Fehlermitteilung gegeben.
 Genaue Beschreibung des Programms s. Skriptum, Kap. 5.5. 

 ACHTUNG: In der Anwendung hat es sich als praktisch erwiesen, 
          dass der Name der Funktion, deren Nullstellen INTSCH ermitteln
	  soll, vom aufrufenden Programm frei gewaehlt werden kann.
	  Dies ist mit der hier vorliegenden Version von INTSCH
	  moeglich. Wie man sieht, lautet der erste Uebergabeparameter

	     double (*fct)(double)     : das bedeutet, dass in INTSCH
	  ueberall dort, wo die Funktion aufgerufen wird, anstelle des
	  "Platzhalters" fct jener Funktionsname verwendet wird, der
	  im entsprechenden Aufrufbefehl von INTSCH (im uebergeordneten
	  Programm) definiert wird, und zwar in der Form

	     &aktueller_name

	  Selbstverstaendlich muss es dann aber auch eine (double)Funktion 
	  geben, welche diesen Namen hat und die von einem (double)Argument
	  abhaengig ist.
*/ 

{
  int pause;
  double eps,xl,yl,xr,yr,speich,x,y,error;

  eps=1.e-12;
  *anz=0;
  xl=anf;
  yl=(*fct)(xl);

  do {
    xr=xl+h;
    yr=(*fct)(xr);

    if(yl*yr > 0.0) {
// Grobsuche geht weiter!
      xl=xr;
      yl=yr;
    }
    else {
// Grobsuche findet Vorzeichenwechsel; Int.schachtelung beginnt!
      *anz=(*anz)+1;
      if(*anz > anzmax) {
        printf("ERR: Too many zeros\n");
        return;
      }
      else {
        do {
          speich=xr;
          x=(xl+xr)/2.0;
          y=(*fct)(x);

//          if(yl*y < 0.0) {   Korr: 4-11-08
          if(yl*y <= 0.0) {
            xr=x;
            yr=y;
          }
          else {
            xl=x;
            yl=y;
          }
// Genauigkeitsabfrage: i.a. relativ, ausser die Nullstelle
// naehert sich einem sehr kleinen Wert (Betrag < EPS:
// dann absolute Fehlerabfrage:
          if(fabs(x) < eps) error=xr-xl;
          else              error=(xr-xl)/fabs(x);

// Schachtelung wird beendet, wenn Nullstelle auf GEN
// lokalisiert ist:
        }while(error >= gen);    
// Abspeicherung der Nullstelle:
        nullst[*anz]=(xr+xl)/2.0;

// Grobsuche wird fortgesetzt:
        xl=speich;
        yl=(*fct)(xl);
      }
    }
  }while(xl+h < aend);
}
