"""
created by jwenner on 10/27/25 to estimate LCOH of a power tower flat receiver. 
    Adapted from "A comprehensive methodology for the design of solar tower external receivers" by Gentile et al. (2024)
"""
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

class costModelFR():
    def __init__(self, Qdes, Htow, Hrec, Wrec, Wpanel, D_o, th, material, A_Hstats, A_land, N_life, N_repl, Qdot_HTFs, times, P_el_pumps, eta_PBII):
        # main assumptions for total equipment cost assessment. refer to Table 2 in paper for more detailed references
        self.C_rec_eng  =3.89e6         # (euro) a fixed engineering cost
        self.C_rec_ref  =44.62e6        # (euro) cost of reference receiver
        self.D_rec_ref  =20             # (m) reference receiver diameter
        self.H_rec_ref  =25             # (m) reference receiver height
        self.c_compl_ref=2147           # (euros/m2) complexity cost of reference receiver
        self.c_coat_purch=4.94          # (euros/m2) purchase cost of absorber coat
        self.c_coat_app =262.3          # (euros/m2) application costs of absorber coat
        self.c_lab      =56.5           # (euros/hour) labor cost
        self.N_h_nozzles=2              # (hours) required hours to make header nozzles
        self.N_h_welds  =1.5            # (hours) make tube to header weld
        self.N_h_clips  =1              # (hours) clip to tube weld
        self.clip_pitch =2              # (m) maximum clip spacing
        self.C_tow_ref  =26.1e6         # (euros) cost of reference tower
        self.H_tow_ref  =203            # (m) reference tower height
        self.c_helio    =126.1          # (euros/m2) cost of heliostats
        self.c_SI       =13.9           # (euros/m2) cost of site improvements
        self.c_land     =2.3            # (euros/m2) cost of land
        self.Q_des_ref  =910e6          # (W) reference design power
        self.C_pip_ref  =18.9e6         # (euros) reference piping cost
        self.C_pump_ref =6.4e6          # (euros) cost of salt pumps
        self.C_contr_ref=15.0e6         # (euros) reference cost of control system
        self.C_spare_ref=1.2e6          # (euros) reference cost of spare parts
        self.c_cont     =0.07           # contingency cost factor
        self.c_EP       =0.13           # engineering procurement cost factor
        self.d_tilda    =0.064          # nominal discount rate
        self.i_tilda    =0.0175         # inflation rate
        self.N_life     =N_life         # (yrs) plant lifetime
        self.N_repl     =N_repl         # (int) number of panel replacements
        self.N_outage   =10             # (days) assumed length of time required to replace a panel
        self.eta_PBII   =eta_PBII       # second law efficiency
        self.T_HTF_in_K =290+565        # (K) assumed inlet temperature of the power block htf
        self.T_HTF_out_K=565+273.15     # (K) assumed outlet temperature of the htf from the power block

        self.Qdot_HTFs      =Qdot_HTFs      # (W) array of Qdots for every time increment
        self.times          =times          # (hr) corresponding time points for every Qdot 
        self.P_el_pumps     =P_el_pumps  # (W) array of pump powers required for every time increment

        self.mat_costs  ={'740H':82.6, 'A230':76.5, '800H':20}             
        # self.mat_costs  ={'740H':82.6, 'A230':35.4, '800H':20}    # use this for comparing to Kelly paper         
                # (euros/kg) alloy costs from Laporte-Azcue paper
        self.rhos       ={'740H':7774, 'A230':8730, '800H':7944}               
                # (kg/m3) A230,740H found in EES, avg. temp. range 290-865, 800H from specialmetals.com

        ## NPS work is unfinished
        # self.df_NPS     =pd.read_csv('NPS_chart.csv')
        ##

        self.Q_des      =Qdes                               # (W) input design power should be in Watts
        self.H_tow      =Htow                               # (m) actual tower height
        self.W_rec      =Wrec                               # (m) flat receiver width
        self.H_rec      =Hrec                               # (m) receiver height
        self.W_panel    =Wpanel                             # (m) panel width
        self.D_o_tube   =D_o                                # (m) tube outer diameter
        self.th         =th                                 # (m) tube thickness
        self.D_i_tube   =D_o-2*th                           # (m) tube inner diameter
        self.N_t        =int(self.W_panel/self.D_o_tube)    # (-) number of tubes per panel
        self.N_p        =self.W_rec/self.W_panel            # (-) TOTAL number of panels in the receiver
        self.c_metal    =self.mat_costs[material] if material in self.mat_costs else print('material cost unknown!')
        self.rho        =self.rhos[material] if material in self.rhos else print('material density unknown!')
        self.th_mani    =9.53e-3                            # (m) wall thickness for NPS 14, schedule 30. 

        self.D_eq       =self.W_rec/math.pi # (m) the equivalent diameter if cylindrical receiver has same circumference as flat rec.'s width
        self.A_rec      =self.W_rec*self.H_rec  # (m2) receiver area
        self.helio_area =A_Hstats       # (m2) required heliostat area
        self.land_area  =A_land         # (m2) required land area

        ## initialize variables
        self.C_SF       =None           # (euros) total cost of solar field
        self.c_compl    =None           # (euros/m2) complexity cost

        ## series of logic checks
        if self.W_rec % self.W_panel != 0:
            print('warning! panel width results in non integer number of panels / receiver width')


        return 

    def calc_LCOH(self):
        """
        calculates levelised cost of heating on an annual basis using defined cost functions for OPEX and CAPEX
        """
        self.calc_CRF()
        self.calc_QtoHTF()
        self.calc_OPEX()

        self.LCOH =(self.CRF*self.CAPEX + self.OPEX)/self.QtoHTF

        return self.LCOH
        
    def calc_CRF(self):
        """
        calculate capital recovery factor, based on equation 63
        """
        frac        =(self.d_tilda-self.i_tilda)/(1+self.i_tilda)

        self.CRF    =( frac*(1+frac)**self.N_life )/( ((1 + frac)**self.N_life) -1 )

        return

    def calc_OPEX(self):
        """
        calculates operating expenditures, which include a flat rate of the CAPEX and the total cost of panel replacements
        """
        self.calc_CAPEX()
        self.calc_total_rec_replacement_cost()

        self.OPEX   =self.C_repl_total/self.N_life + 0.014*self.CAPEX   # formula inferred from discussion on page 14

        return 

    def calc_total_rec_replacement_cost(self):
        """
        calculates total material cost of all replacements required during design lifetime
        """
        ## calculate the cost of a single replacement
        C_repl_panel      =(self.C_rec/self.N_p)*(1+self.c_cont)*(1+self.c_EP) # formula from pg. 14 discussion 
        ## calculate the total replacement cost
        self.C_repl_total =C_repl_panel*self.N_repl

        return

    def calc_QtoHTF(self):
        """
        calculates annual thermal energy yield. Considers outages and pump energy
        """
        self.eta_PB =self.eta_PBII*(self.T_HTF_out_K-self.T_HTF_in_K)/math.log(self.T_HTF_out_K/self.T_HTF_in_K)

        self.QtoHTF =(1-(self.N_repl*self.N_outage)/(365*self.N_life))*np.sum((self.Qdot_HTFs - self.P_el_pumps/self.eta_PB)*self.times)

        return

    def calc_CAPEX(self):
        """
        calculates capital expenditures, which consider the solar field, manufacturing cost, and material cost
        """
        self.calc_TEC()

        self.CAPEX  =self.TEC*(1+self.c_cont)*(1+self.c_EP)

        return

    def calc_TEC(self):
        """
        calculates the total equipment cost, which is like the CAPEX except for contingency and EP costs
        """

        ## calculate the values that are simply scaled up
        self.C_pip  =self.apply_scaling(self.C_pip_ref)
        self.C_pump =self.apply_scaling(self.C_pump_ref)
        self.C_contr=self.apply_scaling(self.C_contr_ref)
        self.C_spare=self.apply_scaling(self.C_spare_ref)
        ##
        self.calc_SF_cost()
        self.calc_rec_CAPEX()
        self.calc_tower_cost()

        self.TEC    =self.C_rec + self.C_SF + self.C_tow + self.C_pip + self.C_pump + self.C_contr + self.C_spare
        
        return

    def calc_rec_CAPEX(self):
        """
        calculates the total receiver costs related to material, manufacturing, and coating
        """

        self.calc_c_compl()
        self.calc_A_coat()

        C_manufacture       =self.A_rec*(self.c_compl - self.c_compl_ref)
        C_coat              =self.A_coat*(self.c_coat_purch + self.c_coat_app)

        self.C_rec  =self.C_rec_eng + self.C_rec_ref*(self.D_eq/self.D_rec_ref)*((self.H_rec/self.H_rec_ref)**0.6) + C_manufacture + C_coat

        return 

    def calc_c_compl(self):
        """
        calculate the complexity cost, which includes the amount of metal
        """
        self.N_clips        =np.ceil(self.H_rec/self.clip_pitch)

        self.calc_m_metal()
        self.c_compl        =( self.m_metal*self.c_metal + self.N_t*self.N_p*self.c_lab*(2*self.N_h_nozzles + 2*self.N_h_welds + self.N_clips*self.N_h_clips)  )/self.A_rec

        return
    
    def calc_m_metal(self):
        """
        calculate the total mass of metal in the absorber tubes, headers, and panel manifolds
        """
        ## estimate mass of tubes + jumper tubes aka headers
        a_tube_wall         =math.pi*(1/4)*(self.D_o_tube**2 - self.D_i_tube**2)
        vol_tube            =a_tube_wall*self.H_rec
        m_tube              =self.rho*vol_tube
        m_tubes_total       =self.N_p*self.N_t*m_tube

        m_tubes_and_headers =m_tubes_total*1.05     # add 5% for the headers. According to pg. 91 of Kelly report: "Advanced Thermal Storage for Central Receivers with Supercritical Coolants"
        
        ## estimate mass of manifolds
        a_flow_tube         =math.pi*(1/4)*(self.D_i_tube**2)
        self.D_i_mani       =np.sqrt(0.5*(self.N_t*a_flow_tube)*(4/math.pi))   # from pg. 91 of report. Factor of 0.5 because flow splits into two directions when in the manifold
        self.D_o_mani       =self.D_i_mani + 2*self.th_mani
        a_mani_wall         =math.pi*(1/4)*(self.D_o_mani**2 - self.D_i_mani**2)
        vol_mani            =a_mani_wall*self.W_panel
        m_mani              =vol_mani*self.rho
        m_manis             =m_mani*2*self.N_p    # there are two manifolds on each panel in the receiver
        
        ## calculate total receiver mass
        self.m_metal    =m_tubes_and_headers + m_manis

        return

    def calc_A_coat(self):
        """
        calculate the receiver area that requires coating in m2. Assume coating must cover entire tube area
        """
        area_outer_tube =math.pi*self.D_o_tube*self.H_rec   # height of the receiver is also the approximate height of the tubes
        self.A_coat     =area_outer_tube*self.N_t*self.N_p

        return

    def calc_SF_cost(self):
        """
        calculates the total solar field costs relating to heliostats, land use, and site improvements
        """
        self.C_SF   =self.helio_area*self.c_helio + self.land_area*self.c_land + self.land_area*self.c_SI

        return

    def calc_tower_cost(self):
        """
        uses simple relation from Turchi to calculate tower cost relative to a reference
        """
        exp_frac    =0.0113

        self.C_tow  =self.C_tow_ref*((math.exp(exp_frac*self.H_tow))/(math.exp(exp_frac*self.H_tow_ref)))

        return
    
    
    def apply_scaling(self,C_ref):
        """
        equation 61 in the paper - used to scale up the horizontal piping, cold salt pump, control system, and spare parts costs for the design power in use
        ---
        inputs: 
        C_ref   -   (euros) reference cost of whatever element we are trying to scale
        ---
        returns:
        C   -   (euros) scaled up output cost
        """
        C =C_ref*(self.Q_des/self.Q_des_ref)**0.7

        return C
    
    def make_TEC_bar_plot(self):
        TEC_dict               ={}
        TEC_dict['solar_field']=[self.C_SF]
        TEC_dict['tower']      =[self.C_tow]
        TEC_dict['receiver']   =[self.C_rec]
        TEC_dict['piping']     =[self.C_pip]
        TEC_dict['pump']       =[self.C_pump]
        TEC_dict['controls']   =[self.C_contr]
        TEC_dict['spares']     =[self.C_spare]
        TEC_df                 =pd.DataFrame(TEC_dict)

        fig,ax=plt.subplots()
        fontsize=14
        ax.bar(TEC_df.columns, TEC_df.iloc[0])
        ax.set_xlabel('TEC subcategory',fontsize=fontsize)
        ax.set_ylabel('cost (euros)', fontsize=fontsize)
        plt.show()

        return
    
    
if __name__ == "__main__":
    ### to do: 
    ## test code
    ## estimate Qto HTF, times, P_el_pumps
    ## implement NPS?

    times=np.ones(8760)*3600 # that many hours in a year
    Qdot_HTFs=np.ones(8760)*50e6
    P_el_pumps=Qdot_HTFs*0.01

    # # for low temeprature supercritical H20 calc in Bruce Kelly work
    # rec_cost_model=costModelFR(Qdes=812e6, Htow=240, Hrec=34.4, Wrec=67.5, Wpanel=2.82, D_o=18.8e-3, th=2.41e-3, material='A230', A_Hstats=1.729332e6, 
    #                      A_land=3e6, N_life=146, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    
    # ## for base nitrate salt case. used to check unit specialty cost and material cost
    # rec_cost_model=costModelFR(Qdes=812e6, Htow=240, Hrec=25, Wrec=62.8, Wpanel=3.925, D_o=56.1e-3, th=1.45e-3, material='A230', A_Hstats=1.6278e6, 
    #                             A_land=4.5e6, N_life=30, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    # ##

    # ## for comparison to Gentile case, Table 10 Haynes 230 rank 2 design choice
    # rec_cost_model=costModelFR(Qdes=670e6, Htow=180, Hrec=20.1, Wrec=50.58, Wpanel=2.53, D_o=60.3e-3, th=1.65e-3, material='A230', A_Hstats=1.282e6, 
    #                             A_land=1.282e6, N_life=30, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    # ##

    ## for comparison to Gentile case, Table 10 Haynes 230 rank 328 design choice
    rec_cost_model=costModelFR(Qdes=670e6, Htow=180, Hrec=16.4, Wrec=16.4*math.pi, Wpanel=4.088, D_o=73.0e-3, th=1.65e-3, material='A230', A_Hstats=1.262e6, 
                                A_land=1.262e6*1.3, N_life=30, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    ##

    # ## for comparison to Gentile case, Table 9 740H rank 1 design choice
    # rec_cost_model=costModelFR(Qdes=670e6, Htow=180, Hrec=16.1, Wrec=16.1*math.pi, Wpanel=2.412, D_o=60.3e-3, th=1.65e-3, material='740H', A_Hstats=1.262e6, 
    #                             A_land=1.262e6*1.3, N_life=30, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    # ##

    # ## for comparison to Gentile case, Table 11 800H rank 5 design choice
    # rec_cost_model=costModelFR(Qdes=670e6, Htow=180, Hrec=20.8, Wrec=20.8*math.pi, Wpanel=3.0912, D_o=48.3e-3, th=1.65e-3, material='800H', A_Hstats=1.204e6, 
    #                             A_land=1.204e6*1.3, N_life=30, N_repl=0, Qdot_HTFs=Qdot_HTFs, times=times, P_el_pumps=P_el_pumps, eta_PBII=0.45)
    # ##

    LCOH          =rec_cost_model.calc_LCOH()
    rec_cost_model.make_TEC_bar_plot()
    print('LCOH is', LCOH*3600*1e6)