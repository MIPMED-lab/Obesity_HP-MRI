

# -------------------------------------------------------------------- NO TRANSPORTER --------------------------------------------------------------------

function PyruvateHP_Cells!(du, u, p, t)
    # Php, Xhp, P, X = u;
    # k_f1, k_r1, T1_P, T1_X = p;

    PyrO_B, PyrO_O, PyrIn, Lac, Ala = u
    kin_B, kin_O, kpl, kal, T1_P, T1_L, T1_A = p


    du[1] = dPyrO_B = - kin_B * PyrO_B - (PyrO_B/T1_P)
    du[2] = dPyrO_O = (kin_B * PyrO_B) - (kin_O*PyrO_O) - (PyrO_O/T1_P)
    du[3] = dPyrIn = (kin_O*PyrO_O) - (kpl*PyrIn) - (kal*PyrIn) - (PyrIn/T1_P)
    du[4] = dLac = (kpl*PyrIn) - (Lac/T1_L)
    du[5] = dAla = (kal*PyrIn) - (Ala/T1_A)

end





function PyruvateHP_NMR_SolveAll(ts, pD, ivss, samps)


    AllSolTest = zeros(length(samps), 5, length(pD[:, 1])) # Simulation of the system observed
    AllSolTest_Off = Array{Any,1}(undef, length(pD[:, 1])) # Simulation of the system before we obvserve it (considering time offset). First column is the time vector
    AllSolTest_Tog = Array{Any,1}(undef, length(pD[:, 1])) # Previous two together. First column is the time vector. 

    for drawInd in collect(1:length(pD[:, 1]))

        p = pD[1:end-1]
        tau = pD[end]

        # Account for the time offset
        ivss2 = ivss;
        prob_off = ODEProblem(PyruvateHP_Cells!, ivss2, (-tau, 0), p)
        part1_off = DifferentialEquations.solve(prob_off, CVODE_BDF(), reltol=1.0e-9, abstol=1.0e-9)

        ivss2 = part1_off.u[end]

        prob = ODEProblem(PyruvateHP_Cells!, ivss2, (ts[1], ts[end]), p)
        part1 = DifferentialEquations.solve(prob, CVODE_BDF(), reltol=1.0e-9, abstol=1.0e-9, saveat=1)
        

        AllSolTest[:, :, drawInd]

        tmp = zeros(length(part1_off.u), 5)



        for j in 1:5
            AllSolTest[:, j, drawInd] = [part1.u[i][j] for i in 1:length(part1.u)][samps.+1]
            tmp[:, j] = [part1_off.u[i][j] for i in 1:length(part1_off.u)]
        end

        AllSolTest_Off[drawInd] = hcat(part1_off.t, tmp)
        AllSolTest_Tog[drawInd] = vcat(AllSolTest_Off[drawInd][1:end-1, :], hcat(samps, AllSolTest[:, :, drawInd]))

    end

    return AllSolTest, AllSolTest_Off, AllSolTest_Tog

end






function ObjectFunctME(p)

    # Define parameter vectors for each amount of cells (last parameter is a time delay, not used in here)
    pD1 = vcat(p[1:end-1], 0);
    # Define time vector
    t2cor = dat2[1:end-dela,1];
    # Define equaly-spaced time vector
    ts1 = collect(0:t2cor[end]);
    # Define initial value for simulation (use of experimental mean)
    ivss1 = [p[end], 0, 0, 0, 0];
    # Convert sampling vector to integer to extract correct elements from simulation
    samps1 = convert.(Int, t2cor);

    if size(pD1) != 1
        pD1 = transpose(pD1);
    end

    # Simulate
    SimOnTime1, SimOffTime1, SimAll1  = PyruvateHP_NMR_SolveAll(ts1, pD1, ivss1, samps1);

    mtimf = convert(Int,round(length(dat2[:,1])/2));

    mp = maximum(dat2[convert(Int, 1+dela):end,2]);
    ma = maximum(dat2[convert(Int, 1+dela):end,3]);
    ml = maximum(dat2[convert(Int, 1+dela):end,4]);


    mmP = sum(sqrt.(((dat2[convert(Int, 1+dela):mtimf,2]./mp) .- ((SimOnTime1[1:mtimf-dela,2].+SimOnTime1[1:mtimf-dela,3])./mp)).^2))
    mmA = sum(sqrt.(((dat2[convert(Int, 1+dela):mtimf,3]./ma) .- (SimOnTime1[1:mtimf-dela,5]./ma)).^2))
    mmL = sum(sqrt.(((dat2[convert(Int, 1+dela):mtimf,4]./ml) .- (SimOnTime1[1:mtimf-dela,4]./ml)).^2))

    obj = (mmP*mmL*mmA)
        
    return(obj)

end


function EstimateParsMod(pathD, filNam, T1s, redo=false)
    
    if isfile(pathD*"\\"*filNam[1:end-4]*"_resM.jld2") && redo == false
        pp = JLD2.load(pathD*"\\"*filNam[1:end-4]*"_resMVect.jld2")["resMMat"]
    else
        Iters = 1;
        CCs = Array{Any}(undef,Iters);
        Fits = Array{Any}(undef,Iters);
        
        println("****************************************************************")
        println("*              THIS PROCESS WILL TAKE ABOUT 2 MIN!             *")
        println("****************************************************************")

        i=1


        # ini = [0.034875772, 0.030084962, 0.605315953, 0.541688476, T1s[1], T1s[3], T1s[2], maximum(dat2[:,2])*2]
        fitness_progress_history = Array{Tuple{Int, Float64},1}()
        callback = oc -> push!(fitness_progress_history, (num_func_evals(oc), best_fitness(oc)))

        resM = bboptimize(ObjectFunctME; SearchRange = [(0, 1), (0, 1), (0, 1), (0, 1), (8, T1s[1]+T1s[1]*0.2),(8, T1s[3]+T1s[3]*0.2),
                                                        (8, T1s[2]+T1s[2]*0.2), (maximum(dat2[:,2]), maximum(dat2[:,2])*1000)], 
                MaxTime = 60*2, method = :adaptive_de_rand_1_bin, 
                CallbackFunction = callback, CallbackInterval = 0.0)

        CCs[i] = fitness_progress_history;
        Fits[i] = resM;

        # Plot convergence curve
        pl=plot(CCs[1], xaxis=:log, linetype=:step, label = "", xlabel = "Iteration", ylabel = "CCV", color = "red", linewidth=3)

        display(pl)

        parfit = zeros(8, Iters);
        for i in 1:Iters
            parfit[:,i] = best_candidate(Fits[i]);
        end

        pp =parfit

        JLD2.save(pathD*"\\"*filNam[1:end-4]*"_resM.jld2", "Fits",Fits)
        JLD2.save(pathD*"\\"*filNam[1:end-4]*"_resMVect.jld2", 
                "resMMat",parfit, "CCs", CCs)

        
    end

    return(pp)
end




function T1Approx(dat2)

    mP = findfirst(dat2[:,2] .== maximum(dat2[:,2]))[1];
    mA = findfirst(dat2[:,3] .== maximum(dat2[:,3]))[1];
    mL = findfirst(dat2[:,4] .== maximum(dat2[:,4]))[1];

    T1s = zeros(3);

    m(t,p) =  p[1].*exp.(-t.*p[2]);
    p0 = [10000, 0.01];

    # Pyruvate
    t = dat2[:,1][mP:end];
    d = dat2[:,2][mP:end];
    fit = curve_fit(m, t, d, p0);
    pp=plot(t, m(t, coef(fit)), title="Pyruvate, T1 = "*string(1/coef(fit)[2]), label = "")
    scatter!(t, d, label = "")
    T1s[1] = 1/coef(fit)[2];
    display(pp)

    # Alanine
    t = dat2[:,1][mA:end];
    d = dat2[:,3][mA:end];
    fit = curve_fit(m, t, d, p0);
    pp=plot(t, m(t, coef(fit)), title="Alanine, T1 = "*string(1/coef(fit)[2]), label = "")
    scatter!(t, d, label = "")
    T1s[2] = 1/coef(fit)[2];
    display(pp)

    # Lactate
    t = dat2[:,1][mL:end];
    d = dat2[:,4][mL:end];
    fit = curve_fit(m, t, d, p0);
    pp=plot(t, m(t, coef(fit)), title="Lactate, T1 = "*string(1/coef(fit)[2]), label = "")
    scatter!(t, d, label = "")
    T1s[3] = 1/coef(fit)[2];
    display(pp)

    return(T1s)

end



function plotFit(dat2, dela, pp, pathD, filNam)
    
    tsC2 = dat2[1:end-dela,1]
    ts = 0:tsC2[end];
    ivss = [pp[end], 0, 0, 0, 0];
    samps = convert.(Int, tsC2);
    SimOnTime1, SimOffTime1, SimAll1  = PyruvateHP_NMR_SolveAll(ts, vcat(vcat(pp[1:end-1]), 0), ivss, samps);
    SimOnTime1b, SimOffTime1b, SimAll1b  = PyruvateHP_NMR_SolveAll(ts, vcat(vcat(pp[1:end-1]), 0), ivss, convert.(Int, ts));


    pp1 = plot(convert.(Int, ts)./60, SimOnTime1b[:,1,1], linewidth = 5, color = "#821114ff", label = "", title = "Pyr_Out_Blood", grid = false, size = (800,600), 
                margin = 8mm, xguidefontsize=16, yguidefontsize=16, tickfontsize=12, xlabel="time (min)", ylabel = "Pyruvate (A.U.)")

    pp2 = plot(convert.(Int, ts)./60, SimOnTime1b[:,2,1]+SimOnTime1b[:,3,1], linewidth = 5, color = "#118264ff", label = "", title = "Pyr_In_Liver", grid = false, size = (800,600), 
                margin = 8mm, xguidefontsize=16, yguidefontsize=16, tickfontsize=12, xlabel="time (min)", ylabel = "Pyruvate (A.U.)")
        scatter!(tsC2./60, dat2[dela+1:end,2], label = "", color = "#58e9c3ff")

    pp3 = plot(convert.(Int, ts)./60, SimOnTime1b[:,4,1], linewidth = 5, color = "#113b82ff", label = "", title = "Lactate_Cells", grid = false, size = (800,600), 
        margin = 8mm, xguidefontsize=16, yguidefontsize=16, tickfontsize=12, xlabel="time (min)", ylabel = "Lactate (A.U.)")
        scatter!(tsC2./60, dat2[dela+1:end,4], label = "", color = "#558ce9ff")

    pp4 = plot(convert.(Int, ts)./60, SimOnTime1b[:,5,1], linewidth = 5, color = "#821181ff", label = "", title = "Alanine_Cells", grid = false, size = (800,600), 
        margin = 8mm, xguidefontsize=16, yguidefontsize=16, tickfontsize=12, xlabel="time (min)", ylabel = "Alanine (A.U.)")
        scatter!(tsC2./60, dat2[dela+1:end,3], label = "", color = "#e959e8ff")


    PP = plot(pp1,pp2,pp3,pp4, layout=(2,2), size = (1000,750))

    savefig(PP,pathD*"\\"*filNam[1:end-4]*"_FitPlots.png")
    savefig(PP,pathD*"\\"*filNam[1:end-4]*"_FitPlots.svg")

    display(PP)

    pyrSSE = sum(((SimOnTime1[:,2,1]+SimOnTime1[:,3,1]) .- dat2[dela+1:end,2]).^2)
    lacSSE = sum((SimOnTime1[:,4,1] .- dat2[dela+1:end,4]).^2)
    alaSSE = sum((SimOnTime1[:,5,1] .- dat2[dela+1:end,3]).^2)
    SSE = pyrSSE+lacSSE+alaSSE

    mp = maximum(dat2[dela+1:end,2]);
    ma = maximum(dat2[dela+1:end,3]);
    ml = maximum(dat2[dela+1:end,4]);


    mmP = sum(sqrt.(((dat2[dela+1:end,2]./mp) .- ((SimOnTime1[:,2,1]+SimOnTime1[:,3,1])./mp)).^2))
    mmA = sum(sqrt.(((dat2[dela+1:end,3]./ma) .- (SimOnTime1[:,5,1]./ma)).^2))
    mmL = sum(sqrt.(((dat2[dela+1:end,4]./ml) .- (SimOnTime1[:,4,1]./ml)).^2))

    obj = (mmP+mmA+mmL)

    return(SSE, obj)
    
end


function findTTP(dat2, dela, pp, pathD, filNam)

    tsC2 = collect(dat2[1:end-dela,1][1]:dat2[1:end-dela,1][end])
    ts = 0:tsC2[end];
    ivss = [pp[end], 0, 0, 0, 0];
    samps = convert.(Int, tsC2);
    SimOnTime1, SimOffTime1, SimAll1  = PyruvateHP_NMR_SolveAll(ts, transpose(vcat(pp[1:end-1], 0)), ivss, samps);

    dP = SimOnTime1[:,2,1]+SimOnTime1[:,3,1];
    ttpP = tsC2[findfirst(maximum(dP) .== dP)[1]]
    println("Time to peak after injection for Pyruvate: "*string(ttpP)*" s")

    dA = SimOnTime1[:,4,1];
    ttpA = tsC2[findfirst(maximum(dA) .== dA)[1]]
    println("Time to peak after injection for Alanine: "*string(ttpA)*" s")

    dL = SimOnTime1[:,5,1];
    ttpL = tsC2[findfirst(maximum(dL) .== dL)[1]]
    println("Time to peak after injection for Lactate: "*string(ttpL)*" s")

    nmsTTP = ["TTP_Pyr", "TTP_Ala", "TTP_Lac"]
    CSV.write(pathD*"\\TiemToPeak.csv", DataFrame(hcat(nmsTTP, [ttpP, ttpA, ttpL]), :auto));
    
    ttp = " Time to peak after injection for Pyruvate: "*string(ttpP)*" s \n Time to peak after injection for Alanine: "*string(ttpA)*" s  \n Time to peak after injection for Lactate: "*string(ttpL)*" s   \n\n\n\n"*"kin_B = "*string(pp[1])*" \nkin_O = "*string(pp[2])*" \nkpl = "*string(pp[3])*" \nkal = "*string(pp[4])*" \nT1_P = "*string(pp[5])*" \nT1_L = "*string(pp[6])*" \nT1_A = "*string(pp[7])*" \n\nPyrO_B__0 = "*string(pp[8]);

    open(pathD*"\\"*filNam[1:end-4]*"_ParsAndTTP.txt", "w") do file
        write(file, ttp);
    end;

    return(ttpP, ttpA, ttpL)


end