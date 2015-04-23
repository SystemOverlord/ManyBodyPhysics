# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:55:02 2015

@author: olaf
"""
import numpy
from scipy import weave

def eigenvalues(a,b,c,n):
    # a.... Vector der unteren Nebendiagonalelemente
    # b.... Vector der Haubtdiagonalelemente
    # c.... Vector der oberen Nebendiagonalelemente
    # n.... Groesse der Matrix

    # Form der Matrix:
    ####################
    # b1  c1
    # a2  b2  c2
    #     a3  b3  c3
    #         ..  ..  ..
    #             ..  ..  cn-1
    #                 an  bn 

    #INTSCH-Parameter:
    anf = 0     # Beginn des Grobsuch-Bereichs
    aend= 100   # Ende des Grobsuch-Bereichs
    h = 1       # Schrittweite im Grobsuch-Bereich
    gen = 0.001 # rel. Genauigkeit der Eigenwerte        
    anzmax = n  # Maximale Anzahl der zu findenden Nullstellen: Abbruch mit Fehler
    anz = 0     # Anzahl der gefundenen Nullstellen
    nullst = numpy.array([0.0]*n)  # Array für Nullstellen

    # Kommentar zum Zusammensuchen der Übergabeparameter:
    # weave.inline(code,['n','a','b','c','anzmax','nullst','anf','aend','h','gen','anz'])    
    
    code = r'''
    #include <stdio.h>
    #include <math.h>

    ///////////// Globale Variablen bzw. Übergabeparameter ////////////////////
    // int n;
    // double *a,*b,*c;
    // double anf,aend,h,gen;
    // int anzmax;
    // double nullst[];
    // int anz;


    // NICHT GEBRAUCHT:
    //////////// allocate a double vector with subscript range v[nl..nh] ////// 
    //double *dvector(long nl, long nh) 
    //{
    //	double *v;
    //
    //	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
    //	if (!v) nrerror("allocation failure in dvector()");
    //	return v-nl+NR_END;
    //}


    ////////// Hyman-Funktion /////////////////////////////////////////////////
    double hymtri(double lam)
    {    
    // Diese Routine dient zur Auswertung der ’Hyman-Funktion’ fuer
    // tridiagonale Matrizen gemaess Glg. (7.43).
    // Die Groesse der Matrix (n) und die Vektoren a, b und c sind
    // global definiert.
    
        int i;
        double x1,x2,x;
        if(n <= 2) 
        {
            printf("Grad zu klein\n");
            return 0.0;
        }
        else 
        {
            x2=-(b[n]-lam)/a[n];
            x1=-((b[n-1]-lam)*x2 + c[n-1])/a[n-1];
            
            for(i=n-2;i>1;i--)
            {
                x= -((b[i]-lam)*x1 + c[i]*x2)/a[i];
                x2=x1;
                x1=x;
            }
        return (b[1]-lam)*x1 + c[1]*x2;
        }
    }
    
    
    ////////// Nullstellensuche mit Intervall Schachtelungsmethode ////////////
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
                    else  error=(xr-xl)/fabs(x);

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


    ////////// Main: Eigenwerte berechnen /////////////////////////////////////
    intsch(&hymtri,anf,aend,h,gen,anzmax,nullst,&anz);
    

    '''
    
    weave.inline(code,['n','a','b','c','anzmax','nullst','anf','aend','h','gen','anz'])
    
    return nullst, anz
