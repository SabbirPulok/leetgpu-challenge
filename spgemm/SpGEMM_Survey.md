A Systematic Survey of General Sparse Matrix-matrix
Multiplication
JIANHUAGAO,WEIXINGJI,FANGLICHANG,SHIYUHAN,BINGXINWEI,
ZEMINGLIU,andYIZHUOWANG,BeijingInstituteofTechnology,China
GeneralSparseMatrix-MatrixMultiplication(SpGEMM)hasattractedmuchattentionfromresearchersin
graphanalyzing,scientificcomputing,anddeeplearning.Manyoptimizationtechniqueshavebeendevel-
opedfordifferentapplicationsandcomputingarchitecturesoverthepastdecades.Theobjectiveofthisarticle
istoprovideastructuredandcomprehensiveoverviewoftheresearchesonSpGEMM.Existingresearches
havebeengroupedintodifferentcategoriesbasedontargetarchitecturesanddesignchoices.Coveredtop-
ics include typical applications, compression formats, general formulations, key problems and techniques,
architecture-orientedoptimizations,andprogrammingmodels.Therationalesofdifferentalgorithmsareana-
lyzedandsummarized.ThissurveysufficientlyrevealsthelatestprogressofSpGEMMresearchto2021.More-
over,athoroughperformancecomparisonofexistingimplementationsispresented.Basedonourfindings,
wehighlightfutureresearchdirections,whichencouragebetterdesignandimplementationsinlaterstudies.
CCSConcepts:•Mathematicsofcomputing→Computationsonmatrices;•Computingmethodolo-
gies→Sharedmemoryalgorithms;Vector/streamingalgorithms;
244
AdditionalKeyWordsandPhrases:SpGEMM,parallelcomputing,sparsematrix,parallelarchitecture
ACMReferenceformat:
Jianhua Gao, Weixing Ji, Fangli Chang, Shiyu Han, Bingxin Wei, Zeming Liu, and Yizhuo Wang. 2023. A
Systematic Survey of General Sparse Matrix-matrix Multiplication. ACM Comput. Surv. 55, 12, Article 244
(March2023),36pages.
https://doi.org/10.1145/3571157
1 INTRODUCTION
SpGEMMisaspecialcaseof generalmatrixmultiplication(GEMM)whentwoinputmatrices
aresparsematrices.Itisafundamentalandexpensivecomputationalkernelinnumerousscientific
computingapplicationsandgraphalgorithms,suchasalgebraicmultigridsolvers[13,14],triangle
counting[9,33,36,124],multi-sourcebreadth-firstsearching[20,53,116],theshortestpathfinding
[26],coloredintersecting[42,73],andsubgraphsmatching[22,119].Hence,theoptimizationof
SpGEMMhasthepotentialtoimpactawidevarietyofapplications.
To the best of our knowledge, this is the first survey article that overviews the developments
of SpGEMM over the past decades. The goal of this survey is to present a working knowledge
ThisworkissupportedbytheNationalNaturalScienceFoundationofChinaunderGrantNo.61972033.
Authors’address:J.Gao,W.Ji(correspondingauthor),F.Chang,S.Han,B.Wei,Z.Liu,andY.Wang,SchoolofComputer
ScienceandTechnology,BeijingInstituteofTechnology,No.5,SouthStreet,Zhongguancun,HaidianDistrict,Beijing,
100081,China;emails:{gjh,jwx,cfl,bit_hsy,bit_wbx,sakusho,frankwyz}@bit.edu.cn.
Permissiontomakedigitalorhardcopiesofallorpartofthisworkforpersonalorclassroomuseisgrantedwithoutfee
providedthatcopiesarenotmadeordistributedforprofitorcommercialadvantageandthatcopiesbearthisnoticeand
thefullcitationonthefirstpage.CopyrightsforcomponentsofthisworkownedbyothersthanACMmustbehonored.
Abstractingwithcreditispermitted.Tocopyotherwise,orrepublish,topostonserversortoredistributetolists,requires
priorspecificpermissionand/orafee.Requestpermissionsfrompermissions@acm.org.
©2023AssociationforComputingMachinery.
0360-0300/2023/03-ART244$15.00
https://doi.org/10.1145/3571157
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:2 J.Gaoetal.
Fig.1. Yeardistributionofreportedcontributions.
oftheunderlyingtheoryandpracticeofSpGEMMforsolvinglarge-scalescientificproblemsand
provide an overview of the algorithms, data structures, and libraries available. This survey cov-
ersthesparseformats,applicationdomains,challengingproblems,architecture-orientedoptimiza-
tiontechniques,andperformanceevaluationofavailableimplementations.Thisstudyfollowsthe
guidelinesofthesystematicliteraturereviewproposedbyKitchenham[74],whichwasinitially
usedinmedicalsciencebutlatergainedinterestinotherfieldsaswell.Accordingtothethreemain
phases:planning,conducting,andreporting,wehaveformulatedthefollowingresearchquestions:
RQ1: WhataretheapplicationsofSpGEMM,andhowaretheyformulated?
RQ2: WhatisthecurrentstatusofSpGEMMresearch?
RQ3: Howdothestate-of-the-artSpGEMMimplementationsperform?
RQ4: Whatchallengescouldbeinferredfromthecurrentresearcheffort?
RegardingRQ1,thisstudypresentsadetailedintroductiontothreetypicalapplicationsandad-
dresseshowSpGEMMisusedintheseapplications.Thiswillgiveaninsightintotherequirements
of SpGEMM in solving real problems. In RQ2, the study looks at existing techniques that were
proposedinrecentdecadesfromdifferentangles.RegardingRQ3,weperformsomeperformance
evaluationstohaveageneralideaabouttheperformanceoftheseimplementationsonprevailing
hardwareplatforms.Finally,inRQ4,wesummarizethechallengesandfutureresearchdirections
accordingtoourinvestigativeresults.
Tohaveabroadcoverage,weperformasystematicliteraturesurveybyindexingpapersfrom
several popular digital libraries (IEEE Explore Digital Library, ACM Digital Library, Elsevier
ScienceDirect, Springer Digital Library, Google Scholar, Web of Science, DBLP, arXiv) using
the keywords “SpGEMM,” “sparse matrix,” “sparse matrix multiplication,” “sparse matrix-matrix
multiplication.” It is an iterative process, and the keywords are fine-tuned according to the
returnedresultsstep-by-step.Wetrytobroadenthesearchasmuchaspossiblewhilemaintaining
amanageableresultset.Then,wereadthetitlesandabstractsofthesepapersandfinallyinclude
92SpGEMMrelatedpapers,whoseyeardistributionispresentedinFigure1.Itcanbeseenthat
thearticlesinthepastthreeyearshaveshownarapidupwardtrend.
ItisdifficulttogiveasufficientandsystematictaxonomyforclassifyingSpGEMMresearch,be-
causethesametopicmayhavedifferentunderstandingsindifferentcontexts,suchasloadbalance
ofSpGEMMindistributed,shared-memorymulticore,singleGPU,multi-GPU,andCPU+GPU.We
selectseveraltopicsthatarefrequentlycoveredbyexistingpapersandpresentthecorrelationbe-
tweenpapersandtopicsintableorfigureofeachsection.
Therestofthearticleisorganizedasfollows:Section2introducesbackgroundindetail,includ-
ingsymbolnotationusedinthisarticle,somepopularandstate-of-the-artcompressionformats
forsparsematrix,andseveralclassicalapplicationsofSpGEMM.FourdifferentSpGEMMformula-
tionsareintroducedinSection3.InSection4,thediscussionabouttheexistingsolutionsforthree
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:3
Table1. SymbolicRepresentation
Symbol Description Symbol Description
A,B,C AandBareinputmatrices,Cistheoutputmatrix · Innerproductoftwovectors
ai∗ TheithrowofA ⊗ Outerproductoftwovectors
a∗j ThejthcolumnofA × G
or
e
a
ne
m
ra
a
l
tr
m
ix
at
a
r
n
ix
da
m
v
u
e
lt
c
i
t
p
o
l
r
icationoftwomatrices
aij TheentryintheithrowandthejthcolumnofA ∗
v
M
e
u
ct
lt
o
i
r
plicationoftwoscalarsorascalaranda
p,q,r Dimensions:A:p×q,B:q×r,C:p×r ◦ Element-wisemultiplication
Table2. ASummaryofSparseFormatsUsedinExistingContributions
Format Contribution
COO [28,40,57,81,84,100,101,133]
[17,28,29,31,35,41,48,61,62,66,69,75,77,78,80,82–84,86,87,92,94,99,103,110,
CSR
111,120,122,125,129–131,133]
CSC [28,31,77,102,129,131]
DCSC [8,17–21,94,109]
DIA[84],ELL[28,69],DCSR[18],BCSR[15],HNI[93],CFM[126],Bitmap/BitMask[55,
Others
72,95,98,129],RLC[27,63],RIR[113],C2SR[115],Tiledstructure[89]
pivotal problems of SpGEMM computation is presented. In Section 5, architecture-oriented
SpGEMM optimization is introduced. Section 6 gives a comprehensive introduction to the
SpGEMMoptimizationusingdifferentprogrammingmodels.Besides,weconductaperformance
evaluation covering most SpGEMM implementations in Section 7. A discussion about the
challengesandfutureworkofSpGEMMispresentedinSection8andfollowedbyourconclusion
inSection9.
2 BACKGROUND
2.1 Preliminaries
2.1.1 Notation. Inthissection,wefirstgivethesymbolicrepresentationcommonlyusedinthis
article,whichispresentedinTable1.Weuseboldcapitalitalicformatrices,lowercasebolditalic
forvectors,andlowercaseitalicforscalars.
2.1.2 SparseMatrix. Thereisnostrictandformaldefinitionofsparsematrix.Themostpopular
one is given by Wilkinson: sparse matrix is any matrix with enough zeros that it pays to take
advantageofthem[52].AnothercommonquantitativedefinitionisgivenbyBarbierietal.[50],
thatis,amatrixAissparseifitsnumberofnon-zeroentries(referredtoasNNZ)isO(n).
2.1.3 CompressionFormat. Storingmatriceswithadensepatternoftenleadstoalotofuseless
calculationsandredundantstorage,becausetheyusuallyhaveafewnon-zeroentries(referredto
asnon-zeros). Theprevailing solutionistostoreeachsparsematrixwith acompressionformat.
Table2summarizesthecontributionsusingdifferentformats.
COO, CSR, CSC, ELL, and DIA are five basic and popular compression formats. COO is the
plainestformatandstorestherowindex,columnindex,andvalueofeachnon-zeroentryinthree
separatearrays.CSRisthemostextensivelyusedformatinexistingwork.Insteadofstoringrow
indices,CSRstorestherowpointerstothefirstnon-zeroentryperrow.CSCreplacesthecolumn
indices array of COO with column pointers. ELL compacts all non-zeros to the left side. DIA is
specifically designed for diagonal sparse matrices. It stores non-zeros in each diagonal and the
offsetofeachdiagonalfromthemaindiagonal.
Inadditiontotheabovefivebasicformats,somenewsparseformatshavebeenproposedover
thepastyears.Buluçetal.[17,18]proposedoublecompressedsparsecolumn(DCSC),anim-
provedformatbasedonCSC.Itisdesignedforhypersparsematrixbyremovingalltherepetitions
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:4 J.Gaoetal.
in column pointers array. They also present DCSR format [18], which is a row-based dialect of
DCSC. Borsˇtnik et al. [15] design an efficient distributed sparse matrix multiplication algorithm
usingblockedCSR(BCSR)format.Parketal.[93]proposeHuffman-codednon-zeroindica-
tion(HNI)format,whichisabitmap-baseddataencoding.Itusesnon-zeroindicationbit-stream
toreplacerowandcolumnindicesandencodesthestreamwithHuffmancoding.Xieetal.[126]
design compressed feature map (CFM) for efficiently storing sparse feature maps in convo-
lutionneuralnetwork(CNN).InSparTen,proposedbyGondimallaetal.[55],asparsetensor
is encoded into a two tuple of a bit-mask representation and a set of non-zeros. The bit-mask
representationhas1’sforpositionswithnon-zerosand0’sotherwise.Thisbit-maskbasedcom-
pression idea is also used in References [72, 95, 98, 129]. Han et al. [27, 63] use a CSR variation,
run-lengthencoding(RLE),tostoreasparseweightmatrixindeepneuralnetwork(DNN).
Thenewformatstoresavectorvandanequal-sizevectorzforeachcolumnintheweightmatrix,
wherevsavesthenon-zeroweights,andzstoresthenumberofzerosbeforethecorrespondingen-
tryinv.Soltaniyehetal.[113]proposeREAPIntermediateRepresentation(RIR)toincrease
the throughput on FPGAs. It has three parts: shared feature, metadata, and distinct features. To
overcometheinefficientmemoryaccessofCSR,Srivastavaetal.[115]proposeachannelcyclic
sparserow(C2SR)format,whichassignseachmatrixrowtoafixedchannelinacyclicmanner.
InTileSpGEMM,Yuetal.[89]proposeasparsetiledatastructurethatdescribesasparsematrix
usingtwolevelsoftileinformation.
2.2 TypicalApplications
SpGEMMisabasicandcriticalcomponentinmanyapplications.Weintroducethebackgroundof
someapplicationsandhowtheSpGEMMisformulatedandusedintheseapplicationsasfollows.
2.2.1 Multi-source BFS. Breadth-first search (BFS) is a key and fundamental subroutine in
manygraphanalysisalgorithms.ThegoaloftheBFSistotraverseagraphfromagivensource
vertex, which can be performed by a sparse matrix-vector multiplication (SpMV) between
theadjacencymatrixAofagraphG = (V,E) andasparsevectorrepresentingthesourcevertex
[132]. Assume n = |V|, then the size of A is n ×n. Let x be a sparse vector with x = 1 and
i
all other entries being zero, then the 1-hop vertices from source vertexi, denoted asv1, can be
i
derived by the SpMV operation:v1 = A×x. Repeating the operation fromv1, we can receive
i i
the2-hopverticesfromi.Finally,acompleteBFSforthegraphG fromvertexi isyielded[53].In
contrast,Multi-SourceBFS(MS-BFS)runsmultipleindependentBFSsconcurrentlyonthesame
graphfrommultiplesourcevertices,whichcanbeformulatedasSpGEMM.Asimpleexamplefor
MS-BFSispresentedinFigure2.Let{1,2}betwosourcevertices,Aistheadjacencymatrixofthe
graph,andX = (x1,x2)isarectangularmatrixrepresentingthesourcevertices,wherex1andx2
aretwosparsecolumnvectorswithx1 =1andx2 =1,respectively,andallotherentriesbeingzero.
1 2
Then,thesparsematrixB1 representingthe1-hopvertices(denotedasv1)fromsourcevertices
{1,2}isgivenbyB1 =A×X.RepeatthemultiplicationofadjacencymatrixandB1:B2 =A×B1,
wecanderivethe2-hopverticesfrom{1,2}.Finally,wegettheresultsofBFSfromvertices1and
2,whichare{1,3,4,2,5,6}and{2,3,4,1,5,6},respectively.
Many applications run hundreds of BFSs over the same graph. Compared with running BFSs
sequentially,runningmultipleBFSsconcurrentlyinasinglekernelallowsustosharethecompu-
tationbetweendifferentBFSswithoutpayingthesynchronizationcost[53]. Asoneofthemost
expensiveoperationsofMS-BFS,SpGEMMisworthyofmoreattentiontobepaid.
2.2.2 Markov Clustering (MCL). Clustering is one of the unsupervised learning methods for
statisticaldataanalysis,andMCLisoneofthegraphclusteringalgorithmsproposedforbiological
data [109]. Using MCL, the closely connected points are grouped into clusters by performing
random walks on a graph based on Markov chains. LetAdenote a probability matrix, and each
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:5
Fig.2. AnexampleofMS-BFS.VerticesfillinginhorizontalandverticallinesarebeingvisitedbytheBFS
startingfromvertices{1}and{2},respectively,andverticesfillinginleftobliquelineshavebeenvisitedby
theBFSstartingfromvertex{1}or{2}[116].
entryin thematrix is theprobabilityof each pointreaching others.Thesum of each column in
this matrix is 1. There are three steps in the clustering iteration. In the first step of expansion,
theprobabilitymatrixofreachingotherpointsstartingfromanypointafteronestepisgivenby
B = A×A. This not only strengthens the connection between different areas, but also leads to
theconvergenceofprobabilities.Inthesecondstep,allentriessmallerthanagiventhresholdare
pruned.Then,thethirdstepofinflationisrequiredtoweakenthepossibilityoflooselyconnected
points by computing the power of each element in the probability matrix. Next, the matrix is
replacedwiththenewmatrixandusedfornextiterations.Afteranumberofiterations,thepoints
inagrapharegraduallyclusteredintogroups.
Due to the high time and memory overhead of MCL, high-performance MCL (HipMCL)
algorithmisproposedforfastclusteringoflarge-scalenetworksondistributedplatforms[109].
2.2.3 AlgebraicMultigridSolvers. AlgebraicMultigrid(AMG)isamulti-gridmethoddevel-
oped based on Geometric Multigrid (GMG). AMG iteratively solves large and sparse linear
systemAx =bbyautomaticallyconstructingahierarchyofgridsandinter-gridtransferoperators
[16, 49]. Generally, AMG includes two processes: setup and solve. The setup phase constructs
multiple components of multigrid algorithm. The solve phase executes multigrid cycling based
onthesecomponents,andSpMVdominatesthisphase.SpGEMMisanimportantkernelinsetup
phase,whosecriticalstepsarepresentedinAlgorithm1.Itfirstconstructsinterpolationoperator
P based on input matrix A (line 3), then the restriction operator R is the transposition of P
l l l
(line4).Finally,thecoarse-gridsystemA l+1 isconstructedusingGalerkinproduct(line5),which
isimplementedwithtwoSpGEMMs[13,14].Generally,P istallandskinny,andR isshortand
l l
fat.Theirnon-zeros’distributioniscloselyrelatedtotheusedinterpolationalgorithms.
TheseSpGEMMs,takingmorethan80%ofthetotalconstructiontime,arethemostexpensive
components. Moreover, the construction of operators (thus, SpGEMM) is an expensive part in
overallexecution,sinceitmayoccurateverytimestep(fortransientproblems)orevenmultiple
timespertimestep(fornon-linearproblems),makingitimportanttooptimizeSpGEMM[48].
ALGORITHM1:ConstructionofseveralimportantoperatorsinAMGsetupphase[14].
Input:A
Output:A1,...,AL ,P0,...,PL−1
1 A0 ←A;
2 forl =0,...,L−1do
3 P l =interpolation(A l );//Constructionofinterpolationoperator
4 R l =P l T //Constructionofrestrictionoperator
5 A l+1 =R l A l P l ;//Constructionofcoarsesystem:Galerkinproduct
6 end
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:6 J.Gaoetal.
Table3. ASummaryofDifferentSpGEMMFormulations
Formulation Contribution
[3,13,28,29,35,39–41,43,45,46,48,61,62,66,78,80,82–84,86,87,89,94,99,
Row-by-row
110,111,113,115,122,130,131]
Inner-product [2,3,9,15,101,109,126,129]
Outer-product [1–3,13,17,18,31,58,77,84,91,107,114,128,133]
Column-by-column [8,18]
Fig.3. ExamplesoffourSpGEMMformulations.Therows,columns,orintermediatematricesinvolvedin
thecomputationarelabeledwithblackborder[115].
2.2.4 Others. SpGEMM is also one of the most important components for genome assembly
[59,60,108],NoSQLdatabase[32,51,67],trianglecounting[33,123,124],graphcontraction[54],
graph coloring [42, 73], the all pairs shortest path [26], sub-graph [22, 119], cycle detection or
counting[25],andmoleculardynamics[1,121].
3 FORMULATIONS
3.1 Overview
Insparsematrixmultiplication,twosparsematricesthataremultipliedcanbeaccessedeitherby
roworcolumn.ThisderivesfourSpGEMMformulations,whicharerow-by-row(RbR),row-by-
column/inner-product (IP), column-by-row/outer-product (OP), and column-by-column
(CbC).Table3summarizestheexistingworkusingdifferentSpGEMMformulations.Asshownin
thetable,RbRisthemostpopularandfavoredformulationbyresearchers,followedbyOP,thenIP,
andtheleastexploredisCbC.Inthissection,weintroducethecalculationofthefourformulations,
followedbyadiscussionontheadvantagesanddisadvantagesofthefourformulations.
3.2 Row-by-row
RbRformulationisbasedontherow-wisepartitioningoftwoinputmatrices.Eachrowc i∗ofC is
calculatedbysummingtheintermediatemultiplicationresultsofeachnon-zeroentrya ik ofa i∗
andcorrespondingrowb k∗ofB,i.e.,
(cid:2)
c i∗ = a ik ∗b k∗,i =1,2,...,p, (1)
k∈Ii(A)
whereI (A) denotesthesetofcolumnindexesk ofthenon-zerosintheithrowofA.Figure3(a)
i
presentsanexampleillustratingthecomputingpatternofRbRformulation.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:7
3.3 Inner-product
Thisformulationisbasedontherow-wiseandcolumn-wisepartitioningofAandB,respectively.
TheresultmatrixC consistsoftheinnerproductofeachrowofAandeachcolumnofB,i.e.,
(cid:2)
c = a ∗b ,i =1,2,...,p,j =1,2,...,r, (2)
ij ik kj
k∈I(i,j)
whereI(i,j) denotesthesetofindexesk suchthatboththeentriesa andb arenon-zero.An
ik kj
examplethatillustratestheinner-productformulationispresentedinFigure3(b).
3.4 Outer-product
Thisformulationisbasedonthecolumn-wiseandrow-wisepartitioningofinputmatricesAand
B,respectively.TheresultmatrixC iscalculatedbysummingtheouterproductofeachcolumn
a∗i ofAandcorrespondingrowb i∗ofB,i.e.,
(cid:2)q
C = a∗i ⊗b i∗. (3)
i=1
Anexamplethatillustratestheouter-productformulationispresentedinFigure3(c).
3.5 Column-by-column
Thisformulationisbasedonthecolumn-wisepartitioningoftwomatrices,whichissimilartothe
RbRformulation.Eachcolumnc∗j oftheresultmatrixCiscalculatedbysummingtheintermediate
multiplicationresultsofeachnon-zero
(cid:2)
entryb
kj
ofb∗j andcorrespondingcolumna∗k ,i.e.,
c∗j = a∗k ∗b
kj
,j =1,2,...,r, (4)
k∈Ij(B)
whereI (B)denotesthesetofrowindexesk ofthenon-zerosinthejthcolumnofB.Anexample
j
thatillustratesthecolumn-by-columnispresentedinFigure3(d).
3.6 Discussion
Srivastavaetal.[115]comparethedatareuseandon-chipmemoryrequirementoffourSpGEMM
formulations. In their work, data reuse is defined as the ratio of the number of multiply-
accumulate(MAC)performedtothesizeofdatareadfromorwrittentomemory.Theyassume
thatallthreesparsematricesaresquare(N ×N)andhavetheuniformnon-zerodistribution,and
AandBhavethesamenumberofnon-zeroentries.AssumingthattheNNZofA,B,andC,hasa
smalldifference,theirdiscussioncanbeconcludedintwopoints.First,therelationshipbetween
thedatareuseoffourformulationsis:DR < DR =DR < DR .Second,fortheon-chip
IP RbR CbC OP
memory,therelationshipisMEM <MEM =MEM <MEM .
IP RbR CbC OP
Here,weextendthediscussiontothesizeofintermediateresults.Letnnz representtheNNZ
C
inoutputmatrixC.ForRbRSpGEMM,onenon-zeroentryofAandthecorrespondingrowofBare
loadedandmultiplied,generatingavectorofsize nnzC.Therefore,itssizeofintermediateresults
N
thatrequiretobereducedisO(nnz×nnzC).ForIPSpGEMM,adotproductbetweenonerowofA
N2
andonecolumnofBisperformed,generatingonenon-zeroentryofC.Therefore,nointermediate
resultsaregenerated.ForOPSpGEMM,anouterproductbetweenonecolumnofAandonerowof
Bisperformed,generatinganintermediateresultmatrixofC.Itstotalsizeofintermediateresults
isO(N ×nnz ).CbCSpGEMMissimilartoRbRSpGEMM.Therefore,therelationshipbetween
C
thesizeofintermediateresultsisIR <IR =IR <OR .
IP RbR CbC OP
Besides, the four SpGEMM formulations are different in storage format and index matching.
RbRpreferstostoretwoinputmatricesandtheoutputmatrixinrow-majorlayout,suchasCSR.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:8 J.Gaoetal.
Fig.4. GeneralworkflowofSpGEMM.
Onthecontrary,thecolumn-majorlayout,suchasCSC,ispreferredbyCbC.However,IPprefers
tostoretwoinputmatricesinrow-majorandcolumn-majorlayouts,respectively.OPprefersan
oppositestorageformat.ThereisnopreferenceforhowtheresultingmatrixisstoredforIPand
OP.AmongthefourSpGEMMformulations,onlyIPrequiresindexmatching.
Generally, the two most basic operations of SpGEMM are scalar multiplication and addition.
However,theycanalsobecustomizedandredefinedinsomeapplications.Forexample,Selvitopi
etal.[108]andGuidietal.[60]presentacustomsemiringtooverloadmultiplicationandaddition
ofSpGEMMinasimilarproteinsequencesidentificationalgorithm.Somepopularlibraries,suchas
CombBLAS[10],CTF[112],andGraphBLAS[37],supportuser-definedmultiplicationandaddition
onsemirings.Onthataccount,SpGEMMcanbegeneralizedandextendedtomorefields.
4 KEYPROBLEMSANDTECHNIQUES
4.1 Overview
ThetypicalworkflowofSpGEMMispresentedinFigure4,whichhasfivestages,includingsize
prediction, memory allocation, work partition and load balance, numeric multiplication, and re-
sultaccumulation.Thesizepredictionstageaimstopredictthememoryfootprintofresultmatrix
before real execution. The memory allocation stage allocates memory space for result matrix on
target device. The objectiveof the work partition and load balance stage is to design an efficient
algorithmtofullyexploittheperformanceofparallelprocessors.Numericmultiplicationsandpar-
tialadditionsareperformed,andalargenumberofintermediateresultsarealsogeneratedinthis
stage.Resultaccumulationdesirestoreducetheseresultsandcalculatefinalresults.
The multiplication result of two sparse matrices is also a sparse matrix, which requires to be
storedinacompressionformat.InCSR/CSCformat,NNZofasparsematrixdominatesitsmem-
oryfootprint,andthesparsityoftheresultmatrixisalwaysunknowninadvance.However,precise
predictionforthesizeofresultmatrixisalwaysexpensiveinpractice.Moreover,withthepopu-
larizationofmulti/many-coreprocessorsanddistributedsystems,theparallelizationofintensive
computing has become a necessary step for accelerating applications. SpGEMM involves three
sparse matrices, which significantly increases the complexity of the problem. Last but not least,
due to the sparsity and irregular non-zero distribution, designing an efficient accumulator (the
datastructurethatweusetoholdtheintermediateresults)isalsoachallengingtask.
In the following sections, we discuss in detail the approaches to deal with three challenging
problems:sizeprediction,workpartitionandloadbalance,andresultaccumulation.
4.2 SizePrediction
4.2.1 Precise Prediction. SpGEMMalgorithms using precisepredictionusually consistof two
phases:symbolicandnumericphases.Inthesymbolicphase,thepreciseNNZineachrow/column
oftheoutputmatrixiscomputedbasedonrowandcolumnindicesofinputsparsematrices(an
exampleisshowninFigure5).Realvaluesofnon-zerosarecalculatedinthenumericphase.
The implementations of SpGEMM in Kokkos Kernels [99], cuSPARSE [90], MKL [68], and
RMerge[57]aretypicalrepresentativesofthismethod.Besides,existingworks[1,40,41,87]also
exploitthisapproach.Tospeedupthesymbolicphase,Devecietal.[45,46]designagraphcom-
pressiontechniquetocompressthematrixBbypackingitscolumnsasbits.InReference[58],the
authorsestimatethememoryrequirementforC aswellasthenumberofbinsandallocatespace
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:9
Fig.5. Anexampleofpreciseandupper-boundprediction.Soliddotsrepresentnon-zeros.ui representsthe
NNZintheithrowofCpredictedbytheupper-boundmethod.
Fig.6. Thethree-layergraph.GiveninputmatricesAandBof3×3,threenodesv1,v2,v3 atthelayerL1
(cid:8) (cid:8) (cid:8) (cid:8)
representthreerowsofA,andv1 ,v2 ,v3 atthelayerL2representitsthreecolumns.vi isconnectedtovj
onlywhenaij isnon-zero.Then,theproductA×Bcanberepresentedasathree-layergraph.
forglobalbinsinthesymbolicphase.SpECK[92]usessizeinformationcollectedinsymbolicexe-
cutiontoguidetheselectionofaccumulatorsinnumericphase.
4.2.2 ProbabilisticMethod. Cohen[30]transformstheproblemofestimatingNNZintheoutput
matrixtothesizeestimationofreachabilitysetsinadirectedgraphandpresentsaMonteCarlo-
basedalgorithmtoestimatethesizeofreachabilitysets.Thealgorithmcanbedemonstratedusing
a hierarchical structure graph of matrix product, as shown in Figure 6. The algorithm starts by
assigningavectorofsamesize,initializedwithexponential-distributionrandomsamples,toeach
nodeinL .Thevectorofeachnodeinthehigherlayerisequaltothecolumn-wiseminimumof
1
thevectorsofitsneighborsinthelowerlayer.Finally,NNZineachrowofC isestimatedbased
onthevectorofeachnodeinL .Foranytoleratedrelativeerrorϵ > 0,itcancomputean1±ϵ
3
approximation of NNZ in result matrix in time O(n/ϵ2). Then, Amossen et al. [4] improve this
methodtoexpectedtimeO(n)forsomeparticularϵ.Anhetal.[5]utilizeasimilarsizeestimation
technologybasedontherowssamplingofAandB.InReference[30],theauthorsintroducean
algorithmtodeterminethemultiplicationorderofchainproductswithminimalnumberofcalcu-
lationoperations.Reference[107]alsousesthisthree-layergraphrepresentationtoestimatethe
sizeofresultmatrix.
4.2.3 Upper-boundPrediction. Thethirdmethodcomputesanupper-boundNNZintheoutput
matrixandallocatescorrespondingmemoryspace.Themostcommonlyusedmethodistocount
NNZinthecorrespondingrowsofBforeachnon-zeroentryinA.Takingthematricespresented
inFigure5forexample,theupperboundofNNZineachrowoftheoutputmatrixisstoredinthe
arrayU = {u ,u ,u ,u }.ThefirstrowofAhastwonon-zeros,whosecolumnindicesare2and
1 2 3 4
4.Therefore,theupper-boundNNZinthefirstrowofC equalstothesumofNNZinthesecond
andforthrowsofB.TheESCalgorithm[14,34]proposedbyBelletal.isarepresentativeofthis
method.Nagasakaetal.[86]alsocountamaximumofscalarnon-zeromultiplicationsperrowof
theresultmatrix.Then,eachthreadallocatesahashtablebasedonthemaximumandreusesthe
hashtablethroughoutthecomputationbyre-initializingatthebeginningofcalculatingeachrow.
4.2.4 ProgressiveMethod. Thefourthmethod,alsoknownastheprogressivemethod,dynam-
icallyallocatesmemoryasneeded.Itfirstallocatesmemoryofpropersizeandthenstartsmatrix
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:10 J.Gaoetal.
Fig.7. ClassificationofSpGEMMpartition.
multiplication. A larger memory block is re-allocated if the current memory is insufficient. The
implementationofSpGEMMinMatlab[52]isarepresentativeofthismethod.Itfirstguessesthe
sizeandthenallocatesamemoryblockthatislargerbyaconstantfactor(typically1.5)thanthe
currentspaceifmorespaceisrequiredatsomepoint.
LiuandVinter[82,83]proposeahybridmethodthatcalculatestheupper-boundNNZforeach
rowandgroupsallrowsintomultiplebinsaccordingtoNNZ.Themethodallocatesspaceofthe
upper-boundsizeforshortrowsandprogressivelyallocatesspaceforlongrows.InTileSpGEMM,
proposed by Yu et al. [89], it first calls the symbolic implementation in NSparse [87] to get the
sparsetilestructureofC.Then,itusesbinarysearchtofindintersectionsparsetilesfromAand
B.Finally,bitmaskoperationsareusedtocalculatethenumberofnon-zerosofeachtileinC.
4.2.5 Discussion. Ofthefourmethods,precisemethodnotonlysavesthememoryusage,but
alsoenablesthesparsestructureofCtobereusedfordifferentmultiplieswiththesamestructureof
inputmatrices[46,61].Moreover,itpresentssignificantbenefitsingraphanalytics,becausemost
ofthemworkonlyonthesymbolicstructure,nonumericphase[124].However,thecalculationof
twophasesmeansthatitneedstoiteratethroughtheinputmatricestwice,leadingtohighercom-
putationoverheadthanothermethods.Theaccuracyandthespeedofthesecondmethoddepend
ontheprobabilisticalgorithmused,andadditionalmemoryallocationmustbelaunchedwhenthe
estimatefails.Theupper-boundmethodisefficientandeasytoimplement,butitusuallyleadsto
memoryover-allocation.Theprogressivemethodallocatesmemorydynamicallyasneeded,and
additionalmemoryallocationmustalsobelaunchedwhenthefirstallocationfails.Inpractice,the
choiceneedstobemadeaccordingtothediscussedproblems.
4.3 WorkPartitionandLoadBalancing
The mainstream work partition and load balancing methods are presented in Figure 7, and we
introduceanddiscussthesemethodsindetailinthefollowingsections.
4.3.1 BlockPartition. BlockpartitionofSpGEMMcanbecategorizedinto1D,2D,and3Dalgo-
rithmsbasedonhowtheypartitiontheworkamongcomputingunits[8,12].Wefirstintroduce
theworkcubenotationbeforedivingintothedetails,asshowninFigure8(b).Thefront,top,and
sideviewsoftheworkcuberepresentthetwoinputmatricesandtheoutputmatrixpresentedin
Figure8(a),respectively.Theworkcubecanbedividedinto3×3×3voxels.Eachvoxelinsiderep-
resentsthescalarmultiplicationoftwonon-zeros,whicharemappedtocellsofthevoxelinthe
frontandtopviewsandcontributestoitsprojectioninthesideview[11].Basedonthisprojection,
taskpartitionofSpGEMMcanbeseenasthepartitionofworkcubeindifferentdimensions.Next,
weintroduce1D,2D,and3Dpartitionbasedonthisnotation.
1DPartition.1Dpartitiononlydividestheworkcubeinoneofthethreedimensions.Asshown
inFigure9,eachsub-figurerepresentsavariantof1Dpartition.A“layer”oftheworkcubeisas-
signedtoonecomputingunitinallvariations.Figure9(a)dividestheworkcubebyplanesparallel
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:11
Fig.8. Theworkcubeformatrixmultiplication.
Fig.9. Threevariantsof1Dpartition.
tothetopview,referredtoasT-partition.Eachcomputingunitisresponsibleforthemultiplication
ofasetofrowsinAandtheentirematrixB,producingthecorrespondingrowsofC.Similarly,
Figures9(b)and(c)dividetheworkcubebyplanesparalleltothefrontviewandsideview,respec-
tively.WerefertothesetwopartitionvariantsasF-partitionandS-partition.
T-partition. Liu et al. [82] group rows ofC to different bins according to their upper-bound
NNZ to maintain a balanced load. Nagasaka et al. [87] partition rows ofC according to precise
NNZineachrowinnumericphase.Kurtetal.[75]proposetoassignonethreadblocktoafixed
number of rows in A and assign a same number of threads in the block to complete the row-
wise multiplication of each row inAand corresponding entries in B. Deveci et al. [45] propose
a hierarchical and parallel SpGEMM algorithm based on Kokkos library [47]. At the first level,
onekokkos-teamisassignedtocalculateasetofrowsofC.Eachkokkos-thread withintheteam
isresponsibletoproduceasubsetoftheserowsatthesecondlevel.Multiplevector-lanesinone
kokkos-thread areassignedtoperformtherow-wisemultiplicationofeachnon-zeroentryinthe
subsetofAandcorrespondingrowsinBatthethirdlevel.TheyalsoexploreT-partitionbutwith
differenttaskassigningschemesforHPCarchitectures[46].Winteretal.[122]presentapartition
schemeassigningthesameNNZofAtoeachblockwhileignoringtherowboundaries.Insteadof
splittingrowsevenly[48],Lietal.[78]splitthematrixintomultiplerowblocksbasedonNNZ.
To address the input or output reuse problem, Zhang et al. [131] split matrixA into row fibers
anddispatchthemtoprocessingelements(PEs)inaSpGEMMacceleratorGAMMA,andeach
PEthenperformsalinearcombinationofrowfibersofB toproducearowfiberofC.Shivdikar
et al. [110] group multiple rows in a single window and assign one PIUMA (Programmable
IntegratedUnifiedMemoryArchitecture)blocktoawindow.Thesizeofawindowdepends
onthescratchpadsize.
F-partition.Azadetal.[9]assignprocessorstoprocesscolumnsoftheuppertriangularmatrix
intrianglecounting.Linetal.[79]designanarchitectureforSpGEMMonFPGAs,thecomputation
ispartitionedevenlytoallPEs,andeachPEisassignedtocalculatemultiplecolumnsofmatrixC.
S-partition.Basedontheouter-productmultiplication,Devecietal.[43]dividerowsofBinto
blocks so each block can be fitted into the HBM of KNL. This partition also induces a row-wise
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:12 J.Gaoetal.
Fig.10. 2Dand3Dpartition.
partitionofA.Zhangetal.[133]optimizetheouter-productSpGEMMbycompactingnon-zeros
of each row in A to the left so the data locality for both input and output matrices are jointly
optimized.Guetal.[58]developanimprovedESCSpGEMMbasedonouterproduct.Theygroup
theexpandedtriplesintobinstosaturatememorybandwidth.
Others. Ballard et al. [13] present a theoretical and detailed analysis for the communication
costofGalerkintripleproductinthesmoothedaggregationofAMGmethods.Theyconcludethat
therow-by-rowproduct(T-partition)isthebest1Dmethodforthefirsttwomultiplications,and
theouterproduct(S-partition)isthebest1Dmethodforthethirdmultiplication.
MostofSpGEMMcanbeimplementedusing1Dpartition.However,whentheNNZincolumns
orrowsofthematrixAorB increases,thecommunicationoverheadbecomessubstantiallyhuge
indistributedcomputing.Thisindicatesthata“layer”in1Dpartitionisrequiredtobeprocessed
bymorecomputingunits.Therefore,the2Dalgorithmswithfine-grainedpartitionareproposed.
2DPartition.2Dalgorithmsdividetheworkcubeintwoofthethreedimensions,sothereare
also three variants. Figure 10(a) illustrates three partition variants that divide the workcube by
planesparalleltotwoofthethreeviews(topandfrontviews,frontandsideviews,topandside
views),referredtoasTF-partition,FS-partition,andTS-partitionhereafter.InTF-partition,each
computingunitcomputesablockofC.However,inTS-partitionandTF-partition,eachcomputing
unitcomputesablockofintermediateresultsofC.
TF-partition. SpSUMMA was first proposedby Buluç et al. [17, 18] based on dense SUMMA
algorithm [118]. They also present a comparative SpCannon algorithm based on dense Cannon
algorithm[23].Bothalgorithmslogicallyorganizeprocessorsasa2DgridandmapablockofA
andB toeachprocessor.InSpSUMMA,eachprocessorbroadcastsdatatootherprocessorsinthe
samerow/columnofthegridandalsoreceivesdatabroadcastbytheseprocessors.InSpCannon,
processorsexchangedatausingpoint-to-pointcommunication.Inbothalgorithms,eachprocessor
accessesarowblockofAandcolumnblockofBandcalculatesaresultblockofC,whichconforms
withTF-partition.MatrixpartitionsimilartoSUMMAisalsousedbyJinetal.[71].Patwaryetal.
[94]partitionAbyrow,whilepartitionBbycolumnwhenacertainconditionissatisfied.
TS-partition. Deveci et al. [43] partitionAand B by row for GPU. When one partition ofA
cannotfitinGPUfastmemory,column-wisepartitionisappliedforrowstripofA.
2Dalgorithmsperformreasonablywellonafewhundredprocesses.However,asthenumber
of processes increases, the communication cost becomes a bottleneck, and 3D algorithms were
developedtoreducethecommunicationcost.
3D Partition. 3D algorithms divide all three dimensions of the workcube. As shown in
Figure10(b),eachcomputingunitownsablockofAandablockofBtocomputeanintermediate
resultblockofthematrixC.
Azadetal.[8]presentaparallelimplementationofthe3DSpGEMMalgorithm.Hussainetal.
[66]useasimilar3Dpartition,whiledividingeachblockofBintomultiplebatchestomeetthesize
ofavailablememory.Toavoidexcessivecommunicationsandlogisticoperationsof2DSpSUMMA,
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:13
Fig.11. HypergraphandbipartitegraphrepresentationofSpGEMM.
Weberetal.[121]proposeanovelSpGEMMalgorithmMPSM3forthecomputationofthedensity
matrixinelectronicstructuretheory.Allprocessesareorganizedasa3DCartesiantopology,and
eachboxintheworkcubeisassignedtoaspecificprocessintermsofthephysicalpropertiesofthe
problem.Yuetal.[89]dividetwoinputsparsematricesintoanumberof16-by-16tilestoutilize
8-bitunsignedchardatatypeforlocalindices.
4.3.2 GraphPartition. Traditionalblock-basedmatrixpartitionsrarelyconsidertheworkload
(NNZineachblock)assignedtoeachprocessingelement.Forexample,SpSUMMApresumesthat
non-zeros are uniformly and randomly distributed across the row/column, which may not hold
formostsparsematrices.
Hypergraph Partition. The hypergraph representation for C = A × B, denoted as H =
(V,N) = (VAB ∪VC,N), is defined as a set of vertices V and a set of nets (hyperedges) N.
Here,wetaketheouter-productSpGEMMasanexampletointroducethehypergraph-basedparti-
tion.InH = (VAB∪VC,N),eachvertexv
i
inVAB denotestheouterproductofthecolumna∗i
ofAwiththerowb i∗ofB.VC containsavertexv
ij
foreachnon-zeroentryc
ij
inC.N containsa
netn foreachnon-zeroentryc inC.Figure11(a)showsanexampleofSpGEMM,andFigure11
ij ij
(b)showsitshypergraphrepresentation.Thetaskofhypergraphpartitioningistodivideahyper-
graphintotwoormoreroughlyequal-sizedpartssuchthatacostfunctiononthenetsconnecting
verticesindifferentpartsisminimized[24].
BipartiteGraphPartition.AbipartitegraphforC =A×B,denotedasG = (VAB∪NC,E),is
definedastwodisjointsetsofverticesVAB andVC andasetofedgesE.Alsotakeouter-product
SpGEMM as an example. The semantics of each vertex in G is the same with those in H. The
differenceisthatthedependenceofc iscapturedwithedges,insteadofanet.Ifanouterproduct
ij
represented byv produces a partial result forc , then an edge connecting verticesv andv
x ij x ij
is added in E. An example of a bipartite graph for the outer-product SpGEMM is presented in
Figure11(c).
Hypergraph-basedpartitionwasfirstappliedinouter-productSpGEMMbyAkbudaketal.[1].
Besides,theyalsopresenttwoextendedhypergraphmodel,whichpartitionsthematrixC byrow
and column, respectively. Ballard et al. [12] present a fine-grained hypergraph model, in which
eachvertexiseitherascalarmultiplicationoranon-zeroentry.Basedonthishypergraphmodel,
Kurtetal.[75]useaniterativemethodtoconstructhypergraphpartitionstoensurethatthedata
entriesinvolvedineachpartitiondonotexceedthecachecapacity.Akbudaketal.[2]usehyper-
graphmodeltoimproveouter-productandinner-productSpGEMM.
Akbudaketal.[3]proposeandcomparemultiplecomputationalpartitioningmodelsbasedon
hypergraphandbipartitegraphpartitioning,withtheaimofreducingmessagevolume.Further,
three communication hypergraph partitioning models for three SpGEMM formulations are pro-
posedtoreducethelatencycost.Thebipartitegraphmodelforrow-by-rowSpGEMMwaslater
used by Demirci et al. [40]. Instead of associating two weights with each vertex, they propose
athree-constrainedpartitioninginwhicheachvertexisassociatedwiththreeweights.Selvitopi
etal.[107]applybipartitegraphandhypergraphmodelsforsimultaneousschedulingofthemap
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:14 J.Gaoetal.
Fig.12. Classificationofaccumulators.
Fig.13. Anexamplefordenseaccumulator.Eachnon-zeroentryinthefirstrowofAismultipliedbythe
correspondingrowofB.Theintermediateresultsarestoredinthedensearrayval,thecorrespondingentry
inthearrayboolissettotrue,andthecolumnindexthathasnotbeenrecordedispushedintothearraycol.
andreducetasksforMapReducejobs.Demircietal.[39]proposehypergraphmodelsfor2Dand
3Dpartitioningtoimprovetheperformanceof2Dand3DSpGEMM.
4.3.3 Discussion. Each partition algorithm usually has its own design considerations and ad-
vantages.Asthemostcommonlyusedsparseformats,CSRandCSCusuallystoresparsematrices
by rows and columns, respectively. It is compatible with the 1D partition. The workload of 1D
partitioniscloselyrelatedtothenon-zerosdistributionsinthedividedrowsorcolumnsandthe
non-zerosdistributionsintheothersparsematrix.TakingT-partitionforexample,weassumethat
eachthreadoragroupofthreadsareresponsibleforthemultiplicationofonerowofAandentire
B.IfBisaregularsparsematrixandhasuniformnon-zerosdistribution,thentheNNZineachrow
ofAdeterminestheworkloadofeachcomputingunit.Therefore,thelongestrowofAdominates
theperformance.TheproblemofloadimbalanceismoreseriousandcomplicatedifBisirregular.
2Dand3Dpartitionsarefine-grainedandproduceamorebalancedworkloadthanthe1Dpartition.
Communicationcostisanimportantmetricforthedistributedsystem.Inthe1Dpartition,each
workprocessaccessesanentireinputmatrix(BforT-partition,AforF-partition)oroutputmatrix
(C forS-partition),whichrequiresahugecommunicationoverhead.Differently,2Dand3Dparti-
tionsjustaccesssomerows/columnsorpartialnon-zeros.Thegraphpartitioningalgorithmsbuild
computation and communication graph models to ensure load balance and minimum communi-
cation cost. For SpGEMM with an irregular computation pattern, graph partitioning algorithms
achieveamorebalancedloadandlowercommunicationcostthanblockpartitioningalgorithms,
butatthecostofhigherpartitioningoverhead.
4.4 ResultAccumulating
Accordingtothestoragestructureoftheintermediateresults,weclassifyaccumulatorsintothree
types:dense,sparse,andhybridaccumulators.TherelatedresearchesareshowninFigure12.
4.4.1 DenseAccumulator. Denseaccumulatorsusedensevectorstocachetheintermediatere-
sultsofcurrent“active”columnsorrowsintheresultmatrix.Themostpopulardenseaccumulator,
firstproposedbyGustavson[61],usuallyhasthreevectors.Thefirstonestoresrealvalues.The
secondoneisusedasatagarraytocheckifacolumnindexhasbeeninsertedbefore.Thethird
onestorescolumnindexes.AnexampleofthedenseaccumulatorispresentedinFigure13.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:15
Fig.14. Anexampleofhash-basedaccumulatorforcalculatingthefirstrowoftheoutputmatrixC.Thehash
tableisinitiallysetto−1.ThecolumnindicesofBserveasthekeyofhashtable.
Gustavson’s algorithm allows random access to a single entry in a specified time, and it is
widelyusedandimprovedbymanysuccessiveresearches[52,77,94,111].Forexample,theSPA
accumulatorinMatlab[52]usesadensevectorofthesamesizeasthecolumnsizeofC,adense
vectorwithtrue/false“occupied”flags,andanunorderedlistoftheindiceswhose“occupied”flags
aretrue.Elliottetal.[48]notethatprolongationmatricesintheAMGmethodareusuallytalland
skinny, so they use a local dense lookup array for each thread that supports fast memory space
allocationforresultmatrixbypage-alignedandsize-trackedtechniques.
4.4.2 SparseAccumulator. Insparseaccumulator,intermediateresultsforarowarestoredin
compactdatastructures.Therearetwotypesofmethodsformerging:listandhash-basedmethods.
List-basedaccumulatingusuallysortstheintermediateresultsbycolumnindexes.According
to the sorting algorithms utilized, we classify researches as follows: Radix sort-based methods
allocate the entries to be sorted to some “buckets” to achieve the goal of sorting. Bell et al. [14]
propose the ESC algorithm, which sorts many scalar products in the second phase. It has been
improvedinReferences[78,83].Daltonetal.[35]usetheB40Cradixsortalgorithm[85],which
allowsspecificationsinthenumberandlocationofthesortingbitstoacceleratethesortoperation.
Guetal.[58]useanin-placeradixsorttogroupkeysbyasinglebytesharingthesamevalidbyte
position.InReference[80],asparseaccumulator,whichsortsdataintheGPUregisters,isproposed.
Mergesort-basedmethodsfirstdividethesequencetobesortedintoseveralsubsequences.Then,
eachsubsequenceisordered,andallsubsequencesarecombinedintoanoverallorderedsequence.
InReference[57],thesparserowsofBselectedandweightedbycorrespondingnon-zerosofAare
mergedinahierarchicalwaysimilartomergesort.Besides,Liuetal.[80]proposeaGPUregister-
basedmergealgorithm,whichdemonstratessignificantspeedupoveritsoriginalimplementation.
Heapsort-basedaccumulatorisproposedbyAzadetal.[8].Theintermediateresultisrepresented
asalistoftriples,andeachtripleincludesrowindex,columnindex,andvalueofeachnon-zero
entry. Ak-way merge onk lists of triples is performed by maintaining a heap of sizek, which
storesthecurrentminimum entryineachtriplelist.Then,amulti-waymergeroutinefindsthe
minimumtriplefromtheheapandmergesitintothecurrentresult.Nagasakaetal.[86]implement
aheap-sortbasedsparseaccumulatoronmulticorearchitectures.
Hash-based.Hash-basedsparseaccumulatorfirstallocatesamemoryspacebasedontheupper
boundestimationasthehashtableandusesthecolumnindexesoftheintermediateresultsasthe
key.Then,thehashtableisrequiredtobeshrunktoadensestate.Finally,wesortthevaluesof
eachrowoftheresultmatrixaccordingtotheircolumnindexestoobtainthefinalresultmatrix
compressedwithsparseformat[80].AnexampleisshowninFigure14.SpGEMMimplementation
incuSPARSE[41,90]isarepresentativeofthisaccumulator.Devecietal.[44]designahashmap-
based, two-level, sparseaccumulatorthat supportsparallel insertionsand merges from multiple
vectorlanes.Weberetal.[121]usethehashtabletostoreeachrowofthematrixC basedonthe
distributedBCSRformat.InReference[5],therowandcolumnindicesofanintermediateresult,
normally stored in two 32-bit numbers, are first compacted into a single 32-bit value and then
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:16 J.Gaoetal.
insertedintohashtableaskey.Itishelpfultoacceleratethelatersortingoperation.Nagasakaetal.
[87]utilizevectorregistersinKNLandmulti-corearchitecturesforhashprobingtoacceleratehash-
basedSpGEMM.Theirhashfunctionisdefinedasthereminderafterdivisionofthecolumnindex
multipliedbyaconstantnumberbyhashtablesize.Liuetal.[80]optimizethedataallocationsof
eachthreadinahash-basedaccumulatorutilizingGPUregisters.Selvitopietal.[109]comparethe
heapandhashsortindistributedHipMCLandconcludethatthelattermethodperformsbetter.
4.4.3 HybridAccumulator. Devecietal.developKKMEM[45]andKKSpGEMM[46]algorithms,
which support two hash-based accumulators and one dense accumulator. Which accumulator
to use depends on the features of the discussed matrix. KKTri-Cilk algorithm [127] uses a
similaraccumulatortothatintheKKSpGEMMalgorithm.Meanwhile,itisoptimizedintheCilk
implementation.
Sparseordenseaccumulatorsareselectedaccordingtothedistributionofnon-zerosindifferent
partsofamatrix.Liuetal.[82,83]proposeahybridparallelresultaccumulatingmethod.Three
sort-basedaccumulatorsareselectedaccordingtotheupper-boundNNZineachrow.InthespECK
algorithm [92], a local balancer decides which accumulator to use for each block to achieve the
best expected performance from the following three accumulators: direct reference, hashing, or
denseaccumulation.Thehashfunctionofitshash-basedaccumulatormultipliestheelementindex
with a prime number and then divides the result by hash table size. TileSpGEMM [89] uses the
sparseaccumulatorworkingonasparsetileandthedenseaccumulatorworkingonadensetile.
Whetheratileisdenseorsparseisdeterminedbyapresetthreshold.
4.4.4 Discussion. Compared with sparse accumulators, dense accumulator requires a large
memoryspace,especiallyinhighconcurrentscenarios.Theyusuallyrequiretomaintainaprivate
dense accumulator for each thread or thread group, which greatly reduces the scalability of the
algorithm.Therearethreecasesinwhichdenseaccumulatorsarepreferred:(1)inhybridaccumu-
lators,toprocesstherowswithalargenumberofnon-zerosperrow;(2)inSpGEMMalgorithms
on CPU, whose large cache and modest computing cores are preferred by dense accumulator;
(3)inSpGEMM,whoseresultmatrixhassmallcolumnsize.Incontrast,thesparseaccumulators
have lower memory requirement and are more suitable for SpGEMM, whose result matrix has
a large number of columns. At the same time, sparse accumulators are also the primary choice
for GPU-based SpGEMM algorithm, because GPU has limited shared memory. Among sparse
accumulators,hash-basedaccumulatorshavelowmemoryspacerequirement,withthecostofhan-
dlingcollisions.Forthreelist-basedsparseaccumulators,theperformanceoftheusedsortingalgo-
rithmdeterminestheirperformance.Heapandmergesorthasthesametimecomplexity,whilethe
latterpresentsahigherspacecomplexity.Radixsortispreferredwhenprocessinglarge-scalearray.
5 ARCHITECTURE-ORIENTEDOPTIMIZATION
Different computer architectures usually have different computing and memory features, so the
optimization of SpGEMM is closely related to the architecture used. In the following sections,
weintroduceSpGEMMoptimizationforfivepopulararchitectures:CPU,GPU,FPGA,ASIC,het-
erogeneous, and distributed platform. Figure 15 summarizes existing contributions on different
architectures.
5.1 CPU-basedOptimization
5.1.1 Optimization for Memory Access. Patwaryetal. [94] proposeto divideAandB byrow
andcolumn,respectively,aimingtoalleviatetheproblemofhighL2cachemissescausedbydense
accumulator.Elliottetal.[48]presentasingle-phaseOpenMPvariantoftheGustavsonalgorithm.
They use page-aligned allocations and track used size and capacity separately to accelerate the
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:17
Fig.15. Asummaryofexistingresearchesondifferentarchitectures.
allocationofthread-localmemories.Guetal.[58]usethepropagationblocktechniquetoimprove
thememorybandwidthutilizationofOPSpGEMM.Theyfirstsavethemultipletuplesgenerated
by the outer product into multiple partially sorted bins and then sort and merge each bin inde-
pendently.Chenetal.[28,29]designmultipleSpGEMMkernelsbasedondifferentsparseformats.
They all utilize coalesced DMA transmission instead of discrete memory access to achieve fast
data loading from main memory. Furthermore, they reserve a minimum partition of B in local
datamemorytoimprovedatareuseandavoidredundantdataloading.
5.1.2 Optimization for Load Balance. In Reference [94], matrices are divided into small parti-
tions and dynamic load balancing is used over the partitions. They find that better results are
achieved when the total number of partitions is 6–10 times the number of threads. Chen et al.
[28,29]presentathree-levelpartitioningschemeforSunwayarchitecture.Atthefirstlevel,each
coregroupintheSW26010processorperformsthemultiplicationofasub-AandB.Atthesecond
level,eachCPEcoreinacoregroupmultipliesthesub-Abyasub-B.Atthethirdlevel,tofitthe
64KBlocaldatamemoryineachCPEcore,sub-Aandsub-B arefurtherpartitionedintoseveral
sets.Toachieveabalancedload,theydivideAandBaccordingtothecomputationalloadsrather
thanrows.
5.1.3 OptimizationforDataStructure. Patwaryetal.[94]storeeachcolumnpartitionofB in
anindividualCSRformat.ThisrequirestochangethedatastructureofBfromCSRtoblockedCSR
inadvance,whichbringsasignificantformatconversionoverhead.Therefore,theyuseasimple
upper-boundmethodtoestimatetheproportionofnon-zerosperrowthatexceedsthesizeofthe
L2cache,andformatconversionoccursonlywhenthisproportionishigherthan30%.
5.2 GPU-basedOptimization
GPUhasemergedasapromisingcomputingdevicetoHPCforitsmassiveparallelismandhigh
memory bandwidth. On the one hand, a GPU has thousands of streaming processors (SP).
ApplicationoptimizationonGPUshouldaddresstheproblemofscalableparallelismandloadun-
balancing.Ontheotherhand,GPUhasahierarchicalmemoryarchitectureincludingthousandsof
registers, on-chip shared memories and caches, and local and global memory. Most GPU
architecture-orientedoptimizationsaimtoreducethememorytrafficbetweenon-chipandglobal
memory.
5.2.1 Optimization for Memory Access. In Reference [35], Dalton et al. propose an improved
ESC method, which stores the non-zeros of A in shared memory to avoid repeated loading
fromglobalmemory.InReference[122],Winteretal.proposeanadaptivechunk-basedGPU
SpGEMM approach (AC-SpGEMM), which improves ESC algorithm by reducing intermedi-
ate results of C within shared memory. Gremse et al. [57] also use shared memory to merge
intermediate results. The main idea is to reduce the overhead of global memory accesses by
mergingrowsusingsub-warps.Then,inReference[87],Nagasakaetal.proposeafastSpGEMM
algorithm only requiring small amount of memory and achieving high performance. Since the
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:18 J.Gaoetal.
memorysubsystemsofNVIDIA® GPUdifferfromgenerationtogenerationsignificantlyinboth
bandwidth and latency, two different data placement methods are proposed in Reference [43].
OneistokeepthepartitionsofAandC infastmemoryandstreamthepartitionsofB toshared
memory.Theotheristheopposite.Whichmethodtochooseoftendependsonthefeaturesofthe
input matrices. In References [82, 83], memory pre-allocation for the result matrix is organized
usingahybridapproachthatsavesalargeamountofglobalmemoryspace.
5.2.2 OptimizationforLoadBalance. InReference[41],onewarpisassignedtoonerowofA,
andeachthreadinthewarpisresponsibleforthemultiplicationofonenon-zeroentryintherow
andthecorrespondingrowofB.Toaddresstheloadimbalancebetweenthreadblocks,Leeetal.
[77] propose Block Reorganizer based on the OP SpGEMM. The Block Reorganizer first gets the
calculationloadofallcolumn-rowpairsofinputmatrices,thencombines,splits,andassignsthese
pairs to thread blocks. Kurt et al. [75] also use one thread block to process multiple rows of A
simultaneously.Yuetal.[89]storeinputmatricesandoutputmatrixasmultiplenon-emptytiles
andassignonewarptoprocessonesparsetileofC inbothsymbolicandnumericphasesonGPU,
whichgreatlyalleviatestheloadimbalancecausedbyirregularnon-zerodistributionsofmatrix
rows.
Toachieveabetterloadbalancing,someworkusesthreadblocktoprocessthesameNNZ.In
Reference[122],Winteretal.splitthenon-zerosofAuniformlyandassigneachthreadblockto
process the same NNZ. However, the workload for each block varies based on the intermediate
productsgeneratedduetotheirregularnon-zerodistributionintherowsofB.Theypresentafine-
grainedloadbalancingstrategywithinthreadblock.Inaddition,someresearchesassigntasksby
intermediateresults.Forexample,theauthorsofReferences[82,83]grouptherowsoftheresult
matrixtomultiplebinsaccordingtoupper-boundNNZestimationofeachrowandassigndifferent
computingunitstoeachbintohaveabetterloadbalancing.
Inrecentyears,someauthorshavesuggestedtwo-levelloadbalancing.SpECK[92]usesboth
globalloadbalance,whichsplitstheworkintoblocks,andlocalloadbalance,whichdecidesthe
numberofthreadsassignedforeachrowofB.Xiaetal.[125]alsoadoptthisloadbalancestrategy
intheirownGPUimplementationofSpGEMM.Also,SpGEMMinKokkosKernels[99]picksthe
best task assignment method according to non-zeros distribution of input matrices. Specifically,
for the matrices with fewer non-zeros per row, it assigns one thread to a row. For the matrices
withmorenon-zeros,multiplethreadsareassignedtoonerow,andvectorparallelismisused.In
Reference[35],SpGEMMisformulatedasalayeredgraphmodel,andtheexpansionphaseofESC
methodisconsideredastheBFSinthelevelsofthelayeredmode.Then,theyparalleltheexpansion
phaseoverthemultiplicationofeachnon-zeroentryofAandthecorrespondingrowofB atthe
granularityofthread,warp,orthreadblock,accordingtothelengthoftherowreferencedfromB.
The emerging tensor core unit (TCU) in GPU attracts the attention of researchers. In
Reference [129], Zachariadis et al. utilize TCUs to improve the performance of SpGEMM. They
partitiontheinputmatricesintotilesandoperateonlyontilesthatcontainoneormorenon-zero
entries.
5.2.3 OptimizationforDataStructure. Liuetal.[82,83]presentefficientparallelmergingmeth-
odsforrowswithalargenumberofintermediateresults.Itconsistsoffoursteps:binarysearchand
duplicateentriesreducing,prefix-sumscan,non-duplicateentriescopy,andmergingoftwosorted
sequenceinonecontinuousmemoryspace.Daltonetal.[35]proposeanoptimizationschemefor
the sorting process of ESC method. They replace global memory-based sorting operations over
a large number of intermediate results with multiple parallel shared memory-based sorting op-
erationsoverasmallnumberofones.Whenthesharedmemoryspaceisinsufficient,theglobal
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:19
memory-basedsortingislaunched.Liuetal.[80]utilizeGPUregisterstooptimizethreetypical
sparseaccumulators:sort,merge,andhash.
5.3 Field-ProgrammableGateArray(FPGA)
FPGAisanalternativecomputingdevicetoCPUandGPU,andithascustomizabledatapaths,flexi-
blememory,andmassiveparallelcomputingunits.Linetal.[79]werethefirsttoaddressSpGEMM
onFPGAs.Theirdesignallowsuser-tunablepower-delayandenergy-delaytradeoffbyemploying
differentnumberofprocessingelementsinarchitecturedesignanddifferentblocksizeinblocking
decomposition.TheauthorsofReference[70]believethatthecomparisonsoftheindexesdomi-
natetheperformanceofSpGEMM.Therefore,theyproposeahighlyparallelarchitecture,which
cancarryoutatleast8×8indexescomparisoninasingleclockcycle.TheirdesignusestheCbC
SpGEMM.Moreover,twodifferentformatsareused,respectively,formatrixA(CSR)andB(CSC).
TheworkofReference[62]proposesanFPGA-basedreconfigurableframeworkFP-AMGthatcan
bereusedforallkernelsinAMG.Theproposedarchitectureisscalableandreconfigurable,butthe
implementationdetailsaboutSpGEMMarenotgiven.
DSPblocksandblockedmemoryareimportantcomponentsinthedesignofSpGEMMonFPGA.
ThenumberofDSPblocksandthesizeofblockedmemorylimitthescaleofthededicatedhard-
ware. The problem size is usually very large in practice, and therefore memory blocks are fre-
quentlyswappedinandoutbecauseofsmallon-chipmemories.However,theoperatingfrequency
ofFPGAgenerallyislowerthanthatoftheCPUorGPU,andthedatatransmissionisalsotime-
consuming.ThesearethetwomainreasonswhyFPGAisnotwidelyemployedforSpGEMM.
5.4 ASIC
OuterSPACE [91] is the first work that addresses the ASIC acceleration of SpGEMM. The
authorsexploreanouter-productbasedmatrixmultiplicationandrevisedsparseformatintheir
implementation. Sriseshan et al. [114] present a memory-centric architecture MetaStrider to
address the fundamental latency-bound inefficiencies of sparse data reduction. SpArch [133] is
proposed to jointly optimize input and output data, which uses outer-product formulation to
reuseinputdataanduseson-chippartialmatrixmergingtoreuseoutputmatrix.Differentfrom
theouterproduct,theauthorsofReference[115]presentanewarchitecture(MatRaptor)basedon
row-wiseproduct.Itaccessesthesparsedatainavectorizedandstreamingfashion.Theworkin
Reference[131]presentsaSpGEMMacceleratorGammathatalsoleveragesGustavsonalgorithm.
It uses an on-chip storage structure FiberCache to capture data reuse patterns and supports
thousandsofconcurrentfinegranularityfibers.Italsoachieveshighthroughputusingadynamic
schedulerandpreprocessingalgorithm.Basedontheobservationthatthemostcompactmemory
formatandthemostefficientcomputingformatneednotbethesame,theauthorsofReference
[97] propose an accelerator extension that supports efficient format conversion and optimal
format-pairprediction.ExTensor[64]isanapproachforacceleratinggeneralizedtensoralgebra
usinghierarchicalandcompositionalintersection.Thereisametadataengineinsidethataggres-
sivelylooksaheadinthecomputationtoremoveuselesscomputationsbeforetheyaredeliveredto
thearithmeticunits.Generallyspeaking,hardwareacceleratorspreferalgorithmsthathavesmall
memoryfootprints.Thereby,row-wiseorinnerandouterjointlyproductionaremorepopularthan
othersincurrentimplementations.Additionally,mostworkreportedareevaluatedusingsoftware
simulators,suchasGEM5[91,115]orhome-madecycle-accuratesimulators[64,97,131,133].
ItisinterestingtoseethatsixoutofsevenpapersofASICoptimizationarereportedafter2020.
However,thereisonlyonepaperthatpresentsdedicatedhardwaredesignforSpGEMM.Thenum-
berofpapersaboutSpGEMMhasrapidlyincreasedinrecentyears.Weareexpectingtoseecus-
tomizedchipscomingoutinthenearfuture.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:20 J.Gaoetal.
5.5 HeterogeneousPlatform
CPU+GPU.Siegeletal.[111]developatask-basedprogrammingmodelandaruntime-supported
execution model to achieve dynamic load balancing on CPU-GPU heterogeneous system. For a
CPU-GPUheterogeneouscomputingsystem,Matametal.[84]proposeseveralheuristicmethods
to identify the proper work division of a subclass of matrices between CPU and GPU. Liu et al.
[83,130]exploitheterogeneousprocessorAMDA10-7850KAPU,whichsharesthesystemRAM
betweenCPUandGPU,toimprovetheperformanceofmemoryre-allocationwhenthefirstsize
predictionfailsforsomematriceswithrelativelylongrows.Besides,inReference[105],Rubensson
etal.presentamethodforparallelblockSpGEMMondistributedmemoryclustersbasedontheir
previouslyproposedChunksandTasksmodel[104].
CPU+FPGA.TheauthorsofReference[113]introduceacooperativeCPU+FPGAarchitecture
REAP.ThesparsematricesarereorganizedonCPUandthenstreamedintoFPGA.TheCPUper-
formsthesymbolicanalysisandpacksthenon-zerosinanintermediaterepresentationtoincrease
regularity.
Heterogeneousmemoryarchitecture.Liuetal.[81]exploretheoptimizationoftwosparse
tensorscontraction(SpTC),ahigh-orderextensionofSpGEMMinessence,usingthepersistent
memory-basedheterogeneousmemory.BasedontheknowledgeoftheSpTCalgorithmanddata
objectcharacteristics,theystaticallyprioritizethedataplacementbetweenDRAMandpersistent
memorymodule(PMM)toachievethebetterperformance.
5.6 DistributedPlatform
ResearchesofSpGEMMonthedistributedplatformusuallyaimatminimizinginter-nodecommu-
nicationandsimultaneouslymaintainingbalancedloadamongmultiplenodes.InReference[71],
Jin et al. propose multiple static and dynamic smart scheduling policies for sparse matrix mul-
tiplicationundertheirproposedsuperprogrammingmodel(SPM).ThedistributedSpGEMM
implementationsprovidedintheEpetraExtandTpetrapackagesofTrilinos[65]dividethefirstma-
trixAintomultiplecomputingnodesbyrow,andtheneachcomputingnodeaccessestheneeded
columnsofthesecondmatrixBthatcorrespondtothecolumndistributionofthedividedArows
fromaremotelocation.
To improve the performance of distributed SpGEMM, Buluç et al. [17] propose two 2D algo-
√rithm√s:SpSUMMAandSpCannon.Forbothalgorithms,P processorsarelogicallyorganizedona
P× Pmesh,andmatricesA,B,andCareassign√edtoprocessorsaccordingto2Ddecomposition.
InSp√Cannon,eachprocessorsendsandreceives P−1point-to-pointmessagesofsizennz(A)/P,
and P −1messagesofsizennz(B)/P.Insteadofthenearest-neighborcommunicationinSpCan-
non,row-wiseandcolumn-wisebroadcastsareusedinSpSUMMA.Figure16showsexamplesof
SpCannon and SpSUMMA. Later, they extend the above algorithms to the distributed platform
withthousandsofprocessorsusingnewerMPIversion[19,21].Acommunicationschemesimilar
toReference[21]isusedinReference[15],whichprovidesadistributedblockedcompressed
sparse row (DBCSR) library aiming to accelerate SpGEMM in the solving of self consistent
field(SCF)equationsfromquantumchemistry.
In the batched 3D SpSUMMA algorithm proposed by Hussain et al. [66], when the memory
requirement to compute the output exceeds available memory of each processor, it accesses
each block of B batch-by-batch. Split-3D-SpGEMM presented by Azad et al. [8] splits each two-
dimensional sub-matrix intoc slices along the third process grid dimension. Rasouli et al. [102]
proposeanewdivide-and-conquerSpGEMM.Itexecutesdatafrompreviousprocessorwhilecom-
municatingitsdatawithneighborstoreducecommunicationtime.Ballardetal.[11]analyzethe
lowerboundsofbandwidthandlatencyofdistributed1D/2D/3Dalgorithmsandproposeimproved
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:21
Fig.16. SpCannonandSpSUMMA.TaketheprocessorP23asanexample.(a)SpCannon:P23sendsA23to
itsleftneighborP22andB23toitsupperneighborP13andreceivesA24fromitsrightneighborP24andB33
fromitslowerneighborP33.(b)SpSUMMA:P23 broadcastsA23 alongthesecondrowandB23 alongthe
thirdcolumn.
algorithmstolowerthecommunicationcost.Weberetal.[121]organizeallprocessesasa3DCarte-
siantopology.EachblockA islocatedonaspecificprocessaccordingtophysicalpropertiesof
ij
thepracticalproblem.TheprocessthatholdsC blockreceivestheblocksA andB fromother
ij ik kj
processesandperformsC =C +A B .
ij ij ik kj
Selvitopietal. [107] usetwo-constrainthypergraphandbipartitegraphmodelstoenablebal-
ancingprocessors’loadsinbothmapandreducephasesandminimizingdatatransferinshuffle
phase.Demircietal.[40]designadistributedSpGEMMalgorithmonAccumulo.Italleviatesmul-
tiple times scanning of the input matrices by using Accumulo’s batch scanning capability. They
alsoproposeabipartitegraph-basedpartitionschemetoreducethetotalcommunicationvolume
andprovideabalanceofworkloadamongservers.InReference[1],Akbudaketal.proposeatwo-
phaseparallelSpGEMMalgorithm,whichusesatwo-constrainthypergraphpartitioningtoguide
partitioning for maintaining a balanced load over two phases. In the first phase, each processor
owns a column stripe of A and a row stripe of B and then finishes the communication-free lo-
calSpGEMMcomputations.Thesecondphasereducespartialresultsyieldedinthefirstphaseto
calculatethefinalvalue.
5.7 Discussion
InviewoftheimportanceofSpGEMMinclassicalscientificcomputinganditswideapplicationin
graphanalysis,ASIC-basedSpGEMMoptimizationhasbeenincreasinginrecentyears.CPU-based
SpGEMMoptimizationworkmainlyfocusesonreasonablepartitionoftheinputmatricestoensure
highcachehitsandloadbalancingamongcomputingunits.TheSpGEMMoptimizationonGPU
usuallyusesregistersandsharedmemorytospeedupmultiplicationandaccumulationasmuchas
possibletoreducetheaccesstoglobalmemory.Inviewofmulti-levelparallelism(thread,warp,and
threadblock),assigningdifferentworkloadstodifferentparallellevelstoensureloadbalancinghas
alsoattractedgreatattentionofresearchers.Onheterogeneousplatforms,anefficientpartitioning
method that can fully utilize the computing power of both CPU and accelerator is one of the
importantdesigningobjectives.Balancedworkloadandminimalcommunicationoverheadamong
computingprocessesarecriticalondistributedplatforms.
6 PROGRAMMINGMODEL
Aprogrammingmodelisabridgebetweenaprogrammer’slogicalviewandphysicalviewofpro-
gramexecutiononsomespecifichardware[7].Overtheyears,anumberofprogrammingmodels
wereproposed,andonlyafewofthemarewidelyused.Mostofthesepopularprogrammingmod-
elsareeitherdrivenbycommercialcompaniesorembracedbyopen-sourcecommunities.Table4
listspopularprogrammingmodelsandframeworksreportedintheliterature.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:22 J.Gaoetal.
Table4. ASummaryofDifferentProgrammingModels
Model Contribution Model Contribution
Tasks-based [71,86,104,105,111,127] OpenCL [84]
OpenMP [39,48,66,84,86,102] MPI [3,11,15,19,21,28,29,48,99,102]
CUDA [14,41,57,77,80,84,87,89,92,125,130] MapReduce [107]
6.1 Task-basedProgrammingModel
Task-basedprogrammingmodelisahigh-levelabstractioninparallelcomputing.Inthismodel,a
workload is partitioned into smaller tasks recursively until a certain task granularity is reached.
Thecomputationoftheprogramfollowsafork-joinmodelthatconcurrenttasksforkatdesignated
pointsandjoinlateratasubsequentpoint.Eachtaskisexecutedononeprocessingunit.
Thereareseveraltask-basedprogrammingmodelsusedinSpGEMMimplementations,andthey
havedifferentparadigms.Cilkisarevised-C/C++languageforparallelcomputing.Itusestwokey-
wordsspawnandsynctoorchestratetasksinthecomputationandusesawork-stealingalgorithm
tobalancetaskintheruntime.Reference[127]proposesKKTri-Clikalgorithmandusesaheuris-
tic strategy to find a balanced partition. Instead of a language extension like Cilk, Threading
BuildingBlocks(TBB)[96]exploresalibrary-basedimplementation.Thelibrarymanagestask
mapping and scheduling at runtime. It is only used for memory allocation/deallocation to gain
higher performance on multi-core and many-core processors in Reference [86]. Reference [111]
definesatask-basedprogrammingframeworkthatsupportspartitioningtheSpGEMMinblocks
to address the problem of load balancing. Different from Reference [111], the authors of Refer-
ence[104]proposeChunksandTasks,anewtask-basedprogrammingmodel.Itmapsthechunks
andtaskstophysicalresources.Reference[105]alsousesthisprogrammingmodelintheirimple-
mentation.TherearealsosomeothercustomizedtasklibrariesforSpGEMM[71].However,these
librariesareonlyusedinthereportedwork.
6.2 OpenMPandMPI
OpenMPisoneofthemostpopularprogrammingparadigmstoenablethreadedparallelism[6].It
usespreprocessingdirectivestotellcompilersthatcodeblockcanbeexecutedinparallel.OpenMP
ismucheasiertousethanthetask-basedmodel.However,programmershavetomakesurethat
theirprogramsusingOpenMParedata-racefree.TheauthorsofReference[86]targetIntelXeon
PhiarchitectureanduseOpenMPtoparallelizeloops.Inthedomainofhigh-performancecomput-
ing,OpenMPisusuallyusedtogetherwithMPI,whichisaspecificationfordistributedcomputing
APIthatenablesmanycomputerstocommunicatewithandworktogether.Generally,SpGEMM
ondistributedsystemsendorsesatwo-levelparallelismthatMPIfacilitatescommunicationamong
SMPnodes,andOpenMPmanagesmultiple-threadsoneachSMPnode[15,39,48,66,102].Almost
alltheworkthatreportedonsupercomputers(CrayXT4[21],BlueGene/Qsystem[3],Sunway
TaihuLight[28,29],AstraandFugaku[99])andclusters[19]useMPI.
OpenMPisalsousedforCPU+Xheterogeneousarchitectures,inwhichXcanbeanyhardware
accelerators,suchasGPU[84]andKNL.Ingeneral,bothhostanddeviceareusedforcomputation
andheuristicsaredesignedtofindthebalancedworkdivisionbetweenCPUandX.
6.3 MapReduce
MapReduce is a structured parallel programming model proposed by Google that serves for
processing large datasets in a massively parallel manner [76]. It is a very important program-
ming pattern that is supported in the Hadoop framework based on the Hadoop file system.
Reference[107]hasitsSpGEMMimplementationatopMR-MPI,anopen-sourceimplementation
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:23
Table5. DetailedInformationofEvaluatedLibraries
Device Library Version Open-source Sizeprediction Accumulator Format
| KK-OpenMP[99] | 3.5.00 | ✓ precise | dense | CSR |
| ------------- | ------ | --------- | ----- | --- |
CPU
✗
| MKL[68] | 2020.4.304 | precise | -   | CSC/CSR/BSR |
| ------- | ---------- | ------- | --- | ----------- |
✗
| cuSPARSE[90] | 11.4.120 | precise       | hash | CSR |
| ------------ | -------- | ------------- | ---- | --- |
| CUSP[34]     | 0.5.0    | ✓ upper-bound | list | COO |
✓
| GPU bhSPARSE[82,83] | 2015.11.6 | hybrid    | hybrid | CSR |
| ------------------- | --------- | --------- | ------ | --- |
| KK-CUDA[99]         | 3.5.00    | ✓ precise | hash   | CSR |
✓
| NSparse[87] | 1.5.0    | precise   | hash   | CSR |
| ----------- | -------- | --------- | ------ | --- |
| spECK[92]   | 2022.1.2 | ✓ precise | hybrid | CSR |
✓
| TileSpGEMM[89] | 2022.1.25 | precise | hybrid | Tiledstructure |
| -------------- | --------- | ------- | ------ | -------------- |
✓
| CTF[112] | 1.5.5 | precise | dense | CSR |
| -------- | ----- | ------- | ----- | --- |
Distributed
| system PETSc | 3.17.3 | ✓ precise | list | CSR |
| ------------ | ------ | --------- | ---- | --- |
ofMapReducewrittenfordistributedmachinesontopofstandardMPI.Thisworkschedulesmap
andreducetasksstaticallyinaMapReducejobtoimprovedatalocalityandloadbalance.
6.4 CUDAandOpenCL
BothCUDAandOpenCLcanbeusedtoprogramGPUdevicestomaximizedataparallelismusing
theSIMTprogrammingmodel.TheirdifferenceliesinthatCUDAisfromNVIDIA® andOpenCL
isanopenstandardsupportingmanydevicesfromdifferentvendors,suchasCPU,GPU,andDSP.
The most used versions in existing work include CUDA 4.0 [14, 84], CUDA 6.0 [57], CUDA 8.0
[80, 125], and CUDA 10.x[92]. CUDA usesmultiple streamsto expressconcurrencyand stream
sequence of operations to GPU devices. Some reported work uses multiple streams to overlap
notonlyexecutions,butalsodatatransfers[84].ResultsreportedinReference[125]confirmthe
effectivenessoflaunchingmultipleCUDAkernelswithCUDAstreamsforeachgrouptoexecute
concurrently.
CUDAandOpenCLfacilitatetheprogrammingofmassivelyparallelcomputingdevices.Never-
theless,adeepunderstandingoftheunderlyingarchitectureisessentialtohavehighperformance
gain.EspeciallyafterthenewVoltaarchitectureisreleased,enablingindependentthreadssched-
uling,anditismorechallengingtowritecorrectcodeonnewGPUdevices.
Inaddition,somelibrariesalsoprovidethePythonwrappersthatbridgethegapbetweenC/C++
andPython.Forexample,Pygraphblas[56]makesiteasyandsimpletocallGraphBLAS[37]APIs
inPython.PyTrilinos[106]allowsPythondeveloperstoimportTrilinos[117]packagesandthen
calltheirAPIsinaPythonprogram.
7 PERFORMANCEEVALUATION
7.1 Overview
Inthissection,weconductaseriesofexperimentstocomparetheSpGEMMperformanceofsev-
eralstate-of-artimplementationsonthreeplatforms,includingCPU,GPU,anddistributedsystem.
Table5listsdetailsofthelibrariesevaluatedineachpart.CPU-andGPU-basedSpGEMMimple-
mentationsinKokkosKernelsarelabeledwithKK-OpenMPandKK-CUDA,respectively.
7.2 SystemSetupandBenchmark
CPU-based SpGEMM is tested on a machine running 64-bit Ubuntu 18.04 and equipped with
one Intel® Xeon® E5-2680 v4 with 2.40 GHz clock frequency and 14 physical cores, supporting
28threads.ForGPU-basedlibraries,twodifferentNVIDIA®GPUsareused.ThefirstisTeslaP100,
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:24 J.Gaoetal.
Table6. TestedSparseMatricesforDistributedSystem
Matrix M N NNZ Matrix M N NNZ
TSOPF_RS_b2052_c1 25,626 25,626 6,761,100 Chebyshev4 68,121 68,121 5,377,761
TSOPF_RS_b678_c2 35,696 35,696 8,781,949 test1 392,908 392,908 12,968,200
largebasis 440,020 440,020 5,560,100 PR02R 161,070 161,070 8,185,136
marine1 400,320 400,320 6,226,538 ohne2 181,343 181,343 11,063,545
Goodwin_127 178,437 178,437 5,778,545 torso1 116,158 116,158 8,516,500
which is based on Pascal architecture and equipped with 3,584 CUDA cores and 16 GB device
memory. The second is Tesla V100, which is based on the Volta architecture and equipped with
5,120 CUDA cores and 16 GB device memory. The version of CUDA Toolkit is 11.4. The perfor-
manceevaluationfordistributedSpGEMMisconductedonaclusterincluding10nodes,eachof
whichisequippedwithtwoIntel®Xeon®Gold6258Rwith2.7GHzclockfrequencyand56physical
cores.
AllthetestedsparsematricesaredownloadedfromtheSuiteSparseMatrixCollection[38].We
usethesamedatasetforperformancetestsonCPUandGPU.Ithas1,880squareand678rectan-
gular matrices. Considering the powerful computing power and non-negligible communication
overhead of distributed system, we choose 10 matrices whose NNZ is greater than 5M. Table 6
listsdetailsofthesematrices.SpGEMMbenchmarkofA×A,commonlyusedinMarkovcluster-
ingalgorithm,isusedforallsquarematrices.ThebenchmarkofA×AT,universalinAMGsolver,
is used for all rectangular matrices. The performance of both single and double floating-point is
tested.
7.3 EvaluationResultsonCPU
MKLusesahighlyencapsulatedinternaldatastructuretoperformoperationsonsparsematrices.
WeencodebothinputmatricesineitherCSRorBSRformatandcallmkl_sparse_sp2mtoperform
SpGEMM.TheMKLimplementationsupportstwo-stageexecution.Therowpointerarrayofthe
outputmatrixiscalculatedinthefirststage.Inthesecondstage,theremainingcolumnindexes
andvaluearraysoftheoutputmatrixarecalculated.Weconsidertheexecutiontimeofthesetwo
stagesastheruntimeofSpGEMMinMKL,whileformatconversionbetweenCSR/BSRandinternal
representation is considered as the preprocessing overhead and discussed later. Besides, we set
28 threads for KK-OpenMP. Figure 17 presents the performance of three SpGEMM algorithms
on CPU ordered by the number of products, which equals to the upper-bound predicted NNZ.
Table7liststhenumberofsparsematricesforwhicheachalgorithmpresentsthebestperformance
andrunssuccessfullyonCPU,aswellastheaveragespeedupofMKL-CSRandMKL-BSRtoKK-
OpenMP.Wemakethefollowingobservations:
• MKL-BSRandMKL-CSRshowbetterperformancethanKK-OpenMPformostsparsematri-
ces and achieve an average speedup of more than 3× and 13× for both single and double
precision,respectively.Specifically,KK-OpenMPachievesthehighestGFLOPSforabout8%
matricesinsingleanddoubleprecision.
• MKL-BSRachievesthehighestGFLOPS formore than900sparsematricesfor bothsingle
anddoubleprecision,butitfailstorunonabout53%matrices.Thereasonisthattheformat
conversionfromdefaultCSRtoBSRposesalargememoryrequirement.
• KK-OpenMPpresentsacomparableperformancetoMKL-CSRandMKL-BSRforlarge-scale
matrices.Thereasonisthatsmall-scalematricescannotfullyutilizethepowerfulhardware,
and their performance gains from multi-threaded parallel computing are offset by thread
creation and synchronization. On the contrary, MKL-CSR runs successfully for more than
90%matricesandshowsthebestperformanceforabouthalfofthem.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:25
Fig.17. GFLOPSonCPUorderedbythetotalnumberofproducts.
Table7. ComparisonoftheNumberofSparseMatricesforwhich
EachAlgorithmAchievestheBestPerformance(theLeftSideof“/”)
andRunsSuccessfully(theRightSideof“/”)onCPU
#ofmatrices SpeeduptoKK-OpenMP
Precision
MKL-CSR MKL-BSR KK-OpenMP MKL-CSR MKL-BSR
single 1,187/2,342 993/1,235 216/2,396 3.46 13.64
double 1,256/2,304 921/1,212 219/2,396 3.56 15.34
WealsoevaluatetheperformanceofsingleprecisiononCPU.ForMKL-CSR,theaverageand
maximum speedups of single to double precision are 1.05× and 2.48×, respectively. MKL-BSR
achievesanaveragespeedupof1.17×,onaverage.Inaddition,anaveragespeedupof1.09×anda
maximumspeedupof2.33×areachievedforKK-OpenMP.
7.4 EvaluationResultsonGPU
Seven GPU-based SpGEMM algorithms are evaluated in this section. Figures 18 and 19 present
theirperformanceonTeslaP100andV100,respectively.Table8liststhenumberofmatricesfor
whicheachalgorithmachievesthebestperformanceandrunssuccessfully.Moreover,theaverage
speeduptoCUSPisgiveninTable9.Wesummarizeourobservationsinthefollowingpoints:
• CUSPandTileSpGEMMrunsuccessfullyforabout72%and68%matricesonbothGPUs.This
is because the SpGEMM implementation of CUSP uses the ESC method, which requires a
largememoryspacetostoretheresultsofscalarmultiplicationandtosupportthesorting
operation. TileSpGEMM uses a tile format, which requires the input matrix to be square.
Therefore,itcannotprocessallrectangularsparsematrices,whichaccountforabout27%
oftheentiredataset.
• OnbothGPUs,CUSPdoesnotpresentthebestperformanceforallmatrices,whilespECK
andNSparseachievethebestperformanceformostsparsematrices.TileSpGEMMoutper-
forms other SpGEMM algorithms for 10%∼20% successfully running matrices. In addition,
cuSPARSE and KK-CUDA achieve the highest GFLOPS for similar number of matrices on
bothGPUs,andbhSPARSEissuperiortootherSpGEMMalgorithmsforafewmatriceson
bothGPUs.
• KK-CUDAshowsdifferentperformanceontwoGPUscomparedwithotherSpGEMMalgo-
rithms.Specifically,itsperformanceonTeslaP100isbetterthanthatofCUSP,anditachieves
anaveragespeeduptoCUSPof1.25×and1.40×insingleanddoubleprecision,respectively.
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:26 J.Gaoetal.
Fig.18. GFLOPSonTeslaP100orderedbythetotalnumberofproducts.
Fig.19. GFLOPSonTeslaV100orderedbythetotalnumberofproducts.
Table8. ComparisonoftheNumberofSparseMatricesforwhichEachAlgorithm
AchievestheBestPerformance(theLeftSideof“/”)andRunsSuccessfully
(theRightSideof“/”)onTeslaP100andV100
SpGEMMalgorithm
GPU Precision
CUSP cuSPARSE NSparse spECK bhSPARSE KK-CUDA TileSpGEMM
single 0/1,848 43/2,347 791/2,352 1,201/2,333 0/2,375 16/2,391 343/1,689
P100
double 0/1,845 33/2,301 798/2,317 1,165/2,281 12/2,383 28/2,388 357/1,689
single 0/1,853 29/2,346 837/2,367 1,337/2,335 2/2,373 19/2,391 170/1,689
V100
double 0/1,848 17/2,305 847/2,378 1,338/2,335 5/2,385 18/2,391 170/1,695
On Tesla V100, however, its overall performance is inferior to that of all other SpGEMM
algorithms.
Wealsocomparetheperformanceofeachalgorithmwithdifferentfloating-pointprecision,and
Table 10 lists the average speedup of single to double precision SpGEMM. We can observe that
thesingle-precisionSpGEMMrunsslightlyfasterthandouble-precisionSpGEMMonbothGPUs.
Specifically,theaveragespeedupsofallSpGEMMalgorithmsfallwithintheinterval[1.00,1.23]
onTeslaP100and[1.02,1.20]onTeslaV100.Althoughsingle-precisionSpGEMMrunsfasterthan
double-precisionSpGEMMonmostsparsematrices,thedifferenceisnotsignificant,especiallyfor
KK-CUDA: Its performance of single precision is very close to that of double precision on both
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:27
Table9. AverageSpeeduptoCUSPonTeslaP100andV100
SpGEMMalgorithm
GPU Precision
cuSPARSE NSparse spECK bhSPARSE KK-CUDA TileSpGEMM
single 3.70 7.06 7.79 2.69 1.25 4.36
P100
double 3.49 7.33 8.02 2.77 1.40 4.85
single 3.25 6.56 7.78 2.66 0.86 5.02
V100
double 2.96 6.56 7.55 2.49 0.88 4.81
Table10. AverageSpeedupofSingletoDoublePrecisiononTeslaP100
andV100
SpGEMMalgorithm
GPU
CUSP cuSPARSE NSparse spECK bhSPARSE KK-CUDA TileSpGEMM
P100 1.09 1.23 1.08 1.11 1.08 1.00 1.03
V100 1.05 1.20 1.06 1.08 1.13 1.02 1.08
GPUs.WefindthatforsomeSpGEMMalgorithmssuchasspECKandKK-CUDA,theimportant
parametersaredeterminedaccordingtothefloatingpointprecisionused.Itmeansthattheper-
formancecomparisonbetweensingleanddoubleprecisionalsoincludestheimpactofparameters
change.However,theideaoftwo-stagecalculationisusedinmostSpGEMMalgorithms.Thefirst
stage calculates the size of the result sparse matrix, which does not involve any floating-point
calculation. When the symbolic phase accounts for a relatively large proportion of the overall
executiontime,thechangeinfloating-pointprecisionhaslessimpactonperformance.
7.5 EvaluationResultsonDistributedSystem
TheperformanceofCTFandPETScondistributedsystemisevaluatedinthissection.Figure20
summarizestheirparallelefficiency,andtheirruntimeiscomparedinTable11,whichliststheav-
eragespeedupto1-processPETSc.ItcanbeobservedthatCTFachieveshigherparallelefficiency
thanPETScinalmostallprocessessettings,butitsaverageperformanceisinferiortothatofPETSc.
Forsingleprecision,theaveragespeedupofPETScwithallprocessessettingsishigherthanthat
ofCTF.Fordoubleprecision,however,theaveragespeedupofCTFisclosetothatofPETScat64,
128,and256processes,andhigherat512processes.Moreover,wefindthatCTFachievesahigher
maximum speedup than PETSc for single-precision SpGEMM when the number of processes is
largerthan32.Fordouble-precisionSpGEMM,itholdsforalltestedsettings.Insummary,PETSc
achievesthebetteraverageperformancethanCTF,butCTFshowsabetterparallelefficiencyand
isexpectedtooutperformPETScwhenalargenumberofprocessesareavailable.Besides,wealso
compare the performance of CTF and PETSc with different floating-point precision. The exper-
imental results from CTF show that the average speedup of single to double over all processes
settingsfallswithintheinterval[1.04,1.11].Theintervalbecomes[1.21,1.40]forPETSc.
7.6 EvaluationResultsofPreprocessingOverhead
Since other SpGEMM algorithms do not has a preprocessing stage, we only present the prepro-
cessing overhead of MKL-CSR, MKL-BSR, and TileSpGEMM in this section. MKL-CSR includes
twopreprocessingoperations:storinganinputCSRmatrixusingtheinternaldatastructure(re-
ferredtoasimport),exportingtheoutputmatrixfromtheinternalstructuretotheCSR(referred
toasexport).AlthoughMKL-BSRissuperiortoMKL-CSRonaconsiderablenumberofmatrices,
italsorequiresextrapreprocessingoperation:formatconvertingbetweenCSRandBSR(referred
toasconvert).InTileSpGEMM,twoinputsparsematricesencodedwiththeCSRarerequiredto
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:28 J.Gaoetal.
Fig.20. Parallelefficiencyondistributedsystem.Eachlineistheaverageparallelefficiencyoverallsparse
matricesdisplayedwitha95%coloredconfidenceinterval.
Table11. AverageSpeedupover1-processPETSconDistributedSystem
SpGEMM PETSc CTF
Processes 2 4 8 16 32 64 128 256 512 1 2 4 8 16 32 64 128 256 512
single 1.39 2.07 3.64 5.91 7.83 10.91 9.81 7.52 7.02 0.29 0.40 0.65 1.27 2.37 4.65 7.37 7.11 4.90 5.36
double 1.40 2.15 3.59 5.74 7.98 10.69 11.06 7.79 6.72 0.34 0.52 0.91 1.59 3.14 6.00 10.41 10.88 7.50 7.72
becompressedusingthetilestructure(referredtoasCSR2Tile),andtheoutputmatrixcompressed
withthetilestructureisalsorequiredtobeconvertedtotheCSR(referredtoasTile2CSR).
Experimental results show that, for both MKL-CSR and MKL-BSR, the overhead of export is
negligible,andimporttakeslessthanoneSpGEMM,onaverage.TheconvertinMKL-BSRismore
time-consumingthanotheroperations,takingabout8SpGEMM,onaverage.InTileSpGEMM,the
formatconversionbetweenthetilestructureandtheCSRrequirescarefulconsideration,especially
CSR2Tile, because it costs tens of single TileSpGEMM for some matrices. Tile2CSR has less time
overheadthanCSR2Tile,about5SpGEMM,onaverage.
From the above discussion, we can conclude that most of the preprocessing overhead of
SpGEMM comes from the format conversion. MKL-BSR, TileSpGEMM, and CUSP use BSR, tile
structure,andCOOformat,respectively.AllotherSpGEMMalgorithmsusethepopularCSR.Our
evaluation on preprocessingoverhead shows that the proposal of a new format requires careful
consideration. It is necessary to evaluate the performance of SpGEMM in specific applications,
simultaneouslyconsideringtheoverheadofformatconversion.
8 CHALLENGESANDFUTUREWORK
SpGEMMhasgainedalotofattentioninrecentdecades,andtheworkhasbeenconductedinmany
directions.Itisexpectedtogetmorein-depthandcomprehensiveresearchandbeappliedtomore
applicationfields.Wesummarizesomepotentialresearchdirectionsandchallengesasfollows:
Optimizationbasedonmachinelearning(ML).OneofthechallengesofSpGEMMisexplor-
ingthesparsityofsparsematrices.Researchersfindthatthenon-zeros’distributiondominatesthe
performanceofSpMVandSpGEMMonthesamearchitecture.Overthepastyears,theparameter
auto-tuningandautomaticselectionofsparseformatsandSpMValgorithmsbasedonMLmodels
havebeendesignedforSpMVoptimization,andsignificantperformanceimprovementhasbeen
observed.However,ML-basedSpGEMMoptimizationismorechallengingduetothesparsitycon-
siderationofthreesparsematrices,complicatedmatrixpartitioning,andloadbalancing.
Optimizationofsizeprediction.Amongthefoursizepredictionmethods,upper-boundpre-
diction may result in memory over-allocation, and progressive and probabilistic prediction may
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:29
leadtomemoryre-allocation.Bothover-andre-allocationaretime-consumingforaccelerationde-
vices,suchasGPU,DSP,FPGA,andTPU.Ononehand,on-devicememorycapacityislimitedand
maynotaccommodatesuchalargeamountofdata.Ontheotherhand,ittakestimetocopydata
to and from device memory because of separated memory space. Precise prediction is the most
popularmethod.Itnotonlyreducestheoverheadofmemorymanagement,butalsoindirectlyim-
provestheperformanceofresultaccumulating.However,wecomparedtheruntimeofeachstage
intwo-stageSpGEMMalgorithmsandfoundthatthesymbolicstageisevenmoreexpensivethan
thenumericalstageforsomematrices.Therefore,moreefficientsizepredictionisrequired.
Heterogeneousarchitecture-orientedoptimization.CPUisgoodatprocessingcomplicated
logic,whiletheGPUisgoodatdensecomputations.Besides,DSPandFPGAmaybeusedindiffer-
entsystems.OneofthecriticalquestionsofportingSpGEMMtotheheterogeneoussystemishow
to achieve load balance and minimize communication traffic. Moreover, sub-matrices may have
differentnon-zeros’distribution.Ideally,relativelydenseblocksshouldbemappedtoacceleration
devices,whileultrasparseblockscanbeassignedtoCPU.Onlyinthiswaycaneachdevicegive
fullplaytoitsarchitecture,optimizingtheoverallcomputingperformanceofSpGEMM.
Application-specific optimization. Application-specific optimization of SpGEMM tends to
bemoreusefulandeffective.Takingunsmoothedaggregation-basedAMGforexample,thesparse
matrixP ofGalerkinproductPTAP isabinarymatrixwithatmostonenon-zeroentryperrow.
Therefore, the library AmgX [88] develops a custom kernel to speed up the product. However,
in classical AMG, P is a tall-skinny sparse matrix. The calculating order of the Galerkin prod-
uct, (PTA)P or PT(AP), is important to the performance of hash accumulator-based SpGEMM.
Specifically,PT(AP) tendstohavelessintermediateresultsaccumulatingoverheadperrowthan
(PTA)P[88].AlthoughtheSpGEMMoptimizationbasedonapplicationcharacteristicsmayreduce
itsextensibility,itisworthwhileifsignificantperformancegainscanbeachieved.Weexpectmore
applicationcharacteristicstobefullyexploitedandutilizedinthefuture.
SpMMiswell-supportedbyexistinghardwareacceleratorsandGPUdevices,asitisanimpor-
tantkernelofmanyconvolutionalalgorithms.However,thesparsityofthematricesinconvolution
ismuchlowerthanthatinHPC.Moreover,mosthigh-performancecomputingunits(suchasten-
sorcores)onlysupportthecalculationoflowfloating-pointprecision(e.g.,16-bithalfprecision),
becausetheprecisionisnotsoimportantinAIapplications.Thisisnottrueforscientificandengi-
neeringcomputing,inwhichdoubleprecisionisusuallyusedfortheconvergenceofsolvers.Con-
vertingasparsematrixtoadenseoneandrunningitontensorcoresisnotpromising,asthereis
toomuchwastingworkandextraoverhead.SpGEMMisusefulintheconvolutionalgorithmifboth
theinputandmodelaresparse.AsthedevelopmentofAIoptimizationtechniquessuchasmodel
pruning,SpGEMMisexpectedtograduallyappearinAIfieldandbeacceleratedbyAIaccelerators.
9 CONCLUSION
ThedesignofanefficientSpGEMMalgorithm,aswellasitsimplementation,iscriticaltomany
large-scalescientificapplications.Therefore,itisnotsurprisingthatSpGEMMhasattractedmuch
attentionfromresearchersovertheyears.Inthissurvey,wehighlightsomedevelopmentsinrecent
yearsandemphasizetheapplications,formulations,challengingproblems,architecture-oriented
optimizations, programming models, and performance evaluation. Some interesting conclusions
canbesummarized.Row-by-rowisthemostcommonlyusedformulationbecauseofitshighpar-
allelismandlowcacherequirement.CSRisthemostfrequentlyusedstorageformat,asitiswidely
used in various fields and thus avoids expensive format conversion. SpGEMM in MKL presents
thebestperformanceonCPU.OnGPU,spECKandNSparseoutperformothersandachievethe
bestperformanceonalargenumberofmatrices.TileSpGEMMalsoshowsexcellentperformance
for regular and square matrices. On distributed system, PETSc is the winner when the number
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:30 J.Gaoetal.
of processes is small. However, CTF presents better performance as the number of processes
increases.
In conclusion, we stress the fact that despite recent progress, there are still important areas
where much work remains to be done, such as different formats supporting, application and
architecture-orientedoptimization,andefficientsizeprediction.However,theheterogeneousar-
chitecture may give rise to new optimization opportunities, and efficient implementation in the
specificarchitectureiswithinreachandcanbeexpectedtocontinueinthefuture.
ACKNOWLEDGMENTS
TheauthorswouldliketothankZhaonianTanandYueyanZhaofortheirearlywork.Wewould
alsoliketoextendourthankstoallreviewersfortheirconstructivecommentsandsuggestions.
REFERENCES
[1] KadirAkbudakandCevdetAykanat.2014.Simultaneousinputandoutputmatrixpartitioningforouter-product-
parallelsparsematrix-matrixmultiplication.SIAMJ.Sci.Comput.36,5(2014),C568–C590.DOI:https://doi.org/10.
1137/13092589X
[2] Kadir Akbudak and Cevdet Aykanat. 2017. Exploiting locality in sparse matrix-matrix multiplication on many-
corearchitectures.IEEETrans.ParallelDistrib.Syst.28,8(2017),2258–2271.DOI:https://doi.org/10.1109/TPDS.2017.
2656893
[3] KadirAkbudak,OguzSelvitopi,andCevdetAykanat.2018.Partitioningmodelsforscalingparallelsparsematrix-
matrixmultiplication.ACMTrans.ParallelComput.4,3(2018),13:1–13:34.DOI:https://doi.org/10.1145/3155292
[4] RasmusResenAmossen,AndreaCampagna,andRasmusPagh.2014.Bettersizeestimationforsparsematrixprod-
ucts.Algorithmica69,3(2014),741–757.DOI:https://doi.org/10.1007/s00453-012-9692-9
[5] PhamNguyenQuangAnh,RuiFan,andYonggangWen.2016.BalancedhashingandefficientGPUsparsegeneral
matrix-matrixmultiplication.InInternationalConferenceonSupercomputing.ACM,36:1–36:12.DOI:https://doi.org/
10.1145/2925426.2926273
[6] OpenMPARB.2021.OpenMP:TheOpenMPAPIspecificationforparallelprogramming.Retrievedfromhttps://
www.openmp.org/.
[7] Krste Asanović, Ras Bodik, Bryan Christopher Catanzaro, Joseph James Gebis, Parry Husbands, Kurt Keutzer,
DavidA.Patterson,WilliamLesterPlishker,JohnShalf,SamuelWebbWilliams,andKatherineA.Yelick.2006.The
LandscapeofParallelComputingResearch:AViewfromBerkeley.TechnicalReportUCB/EECS-2006-183.EECSDepart-
ment,UniversityofCalifornia,Berkeley.Retrievedfromhttp://www2.eecs.berkeley.edu/Pubs/TechRpts/2006/EECS-
2006-183.html.
[8] ArifulAzad,GreyBallard,AydinBuluç,JamesDemmel,LauraGrigori,OdedSchwartz,SivanToledo,andSamuel
Williams.2016.Exploitingmultiplelevelsofparallelisminsparsematrix-matrixmultiplication.SIAMJ.Sci.Comput.
38,6(2016),C624–C651.DOI:https://doi.org/10.1137/15M104253X
[9] ArifulAzad,AydinBuluç,andJohnR.Gilbert.2015.Paralleltrianglecountingandenumerationusingmatrixalge-
bra.InIEEEInternationalParallelandDistributedProcessingSymposiumWorkshop.IEEEComputerSociety,804–811.
DOI:https://doi.org/10.1109/IPDPSW.2015.75
[10] ArifulAzad,OguzSelvitopi,MdTaufiqueHussain,JohnR.Gilbert,andAydinBuluç.2022.CombinatorialBLAS
2.0:Scalingcombinatorialalgorithmsondistributed-memorysystems.IEEETrans.ParallelDistrib.Syst.33,4(2022),
989–1001.DOI:https://doi.org/10.1109/TPDS.2021.3094091
[11] GreyBallard,AydinBuluç,JamesDemmel,LauraGrigori,BenjaminLipshitz,OdedSchwartz,andSivanToledo.2013.
Communicationoptimalparallelmultiplicationofsparserandommatrices.In25thACMSymposiumonParallelism
inAlgorithmsandArchitectures.ACM,222–231.DOI:https://doi.org/10.1145/2486159.2486196
[12] GreyBallard,AlexDruinsky,NicholasKnight,andOdedSchwartz.2016.Hypergraphpartitioningforsparsematrix-
matrixmultiplication.ACMTrans.ParallelComput.3,3(2016),18:1–18:34.DOI:https://doi.org/10.1145/3015144
[13] GreyBallard,ChristopherM.Siefert,andJonathanJ.Hu.2016.Reducingcommunicationcostsforsparsematrix
multiplicationwithinalgebraicmultigrid.SIAMJ.Sci.Comput.38,3(2016),C203–C231.DOI:https://doi.org/10.1137/
15M1028807
[14] NathanBell,StevenDalton,andLukeN.Olson.2012.Exposingfine-grainedparallelisminalgebraicmultigridmeth-
ods.SIAMJ.Scient.Comput.34,4(2012),C123–C152.DOI:https://doi.org/10.1137/110838844
[15] UrbanBorstnik,JoostVandeVondele,ValéryWeber,andJürgHutter.2014.Sparsematrixmultiplication:Thedis-
tributedblock-compressedsparserowlibrary.ParallelComput.40,5-6(2014),47–58.DOI:https://doi.org/10.1016/j.
parco.2014.03.012
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:31
[16] WilliamL.Briggs,VanEmdenHenson,andStephenF.McCormick.2000.AMultigridTutorial,SecondEdition.SIAM.
[17] AydinBuluçandJohnR.Gilbert.2008.Challengesandadvancesinparallelsparsematrix-matrixmultiplication.In
InternationalConferenceonParallelProcessing.IEEEComputerSociety,503–510.DOI:https://doi.org/10.1109/ICPP.
2008.45
[18] AydinBuluçandJohnR.Gilbert.2008.Ontherepresentationandmultiplicationofhypersparsematrices.In22nd
IEEEInternationalSymposiumonParallelandDistributedProcessing.IEEE,1–11.DOI:https://doi.org/10.1109/IPDPS.
2008.4536313
[19] AydinBuluçandJohnR.Gilbert.2010.Highlyparallelsparsematrix-matrixmultiplication.CoRRabs/1006.2183
(2010).
[20] AydinBuluçandJohnR.Gilbert.2011.ThecombinatorialBLAS:Design,implementation,andapplications.Int.J.
HighPerform.Comput.Applic.25,4(2011),496–509.DOI:https://doi.org/10.1177/1094342011403516
[21] AydinBuluçandJohnR.Gilbert.2012.Parallelsparsematrix-matrixmultiplicationandindexing:Implementation
andexperiments.SIAMJ.Sci.Comput.34,4(2012),C170–C191.DOI:https://doi.org/10.1137/110848244
[22] AydinBuluçandKameshMadduri.2011.Parallelbreadth-firstsearchondistributedmemorysystems.InConference
onHighPerformanceComputingNetworking,StorageandAnalysis.ACM,65:1–65:12.DOI:https://doi.org/10.1145/
2063384.2063471
[23] LynnElliotCannon.1969.ACellularComputertoImplementtheKalmanFilterAlgorithm.MontanaStateUniversity.
[24] ÜmitV.Çatalyürek,BoraUçar,andCevdetAykanat.2011.Hypergraphpartitioning.InEncyclopediaofParallelCom-
puting.SpringerUS,Boston,MA,871–881.DOI:https://doi.org/10.1007/978-0-387-09766-4_1
[25] KerenCensor-Hillel,PetteriKaski,JanneH.Korhonen,ChristophLenzen,AmiPaz,andJukkaSuomela.2015.Alge-
braicmethodsinthecongestedclique.InACMSymposiumonPrinciplesofDistributedComputing.ACM,143–152.
DOI:https://doi.org/10.1145/2767386.2767414
[26] TimothyM.Chan.2007.Morealgorithmsforall-pairsshortestpathsinweightedgraphs.In39thAnnualACMSympo-
siumonTheoryofComputing(STOC’07).ACM,NewYork,NY,590–598.DOI:https://doi.org/10.1145/1250790.1250877
[27] Yu-HsinChen,TusharKrishna,JoelS.Emer,andVivienneSze.2017.Eyeriss:Anenergy-efficientreconfigurable
acceleratorfordeepconvolutionalneuralnetworks.IEEEJ.SolidStateCirc.52,1(2017),127–138.DOI:https://doi.
org/10.1109/JSSC.2016.2616357
[28] YuedanChen,KenliLi,WangdongYang,GuoqingXiao,XianghuiXie,andTaoLi.2019.Performance-awaremodel
forsparsematrix-matrixmultiplicationontheSunwayTaihulightsupercomputer.IEEETrans.ParallelDistrib.Syst.
30,4(2019),923–938.DOI:https://doi.org/10.1109/TPDS.2018.2871189
[29] YuedanChen,GuoqingXiao,andWangdongYang.2020.OptimizingpartitionedCSR-basedSpGEMMontheSunway
Taihulight.NeuralComput.Appl.32,10(2020),5571–5582.DOI:https://doi.org/10.1007/s00521-019-04121-z
[30] EdithCohen.1997.Size-estimationframeworkwithapplicationstotransitiveclosureandreachability.J.Comput.
Syst.Sci.55,3(1997),441–453.DOI:https://doi.org/10.1006/jcss.1997.1534
[31] EdithCohen.1998.Structurepredictionandcomputationofsparsematrixproducts.J.Comb.Optim.2,4(1998),
307–332.DOI:https://doi.org/10.1023/A:1009716300509
[32] JeffreyCohen,BrianDolan,MarkDunlap,JosephM.Hellerstein,andCalebWelton.2009.MADskills:Newanalysis
practicesforbigdata.Proc.VLDBEndow.2,2(2009),1481–1492.DOI:https://doi.org/10.14778/1687553.1687576
[33] JonathanD.Cohen.2009.GraphtwiddlinginaMapReduceworld.Comput.Sci.Eng.11,4(2009),29–41.DOI:https:
//doi.org/10.1109/MCSE.2009.120
[34] StevenDalton,NathanBell,LukeOlson,andMichaelGarland.2014.CUSP:GenericParallelAlgorithmsforSparse
MatrixandGraphComputations.Retrievedfromhttp://cusplibrary.github.io/.
[35] StevenDalton,LukeN.Olson,andNathanBell.2015.Optimizingsparsematrix-matrixmultiplicationfortheGPU.
ACMTrans.Math.Softw.41,4(2015),25:1–25:20.DOI:https://doi.org/10.1145/2699470
[36] TimothyA.Davis.2018.GraphalgorithmsviaSuiteSparse:GraphBLAS:trianglecountingandK-truss.InIEEEHigh
PerformanceExtremeComputingConference.IEEE,1–6.DOI:https://doi.org/10.1109/HPEC.2018.8547538
[37] TimothyA.Davis.2019.Algorithm1000:SuiteSparse:GraphBLAS:Graphalgorithmsinthelanguageofsparselinear
algebra.ACMTrans.Math.Softw.45,4(2019),44:1–44:25.DOI:https://doi.org/10.1145/3322125
[38] TimothyA.DavisandYifanHu.2011.TheUniversityofFloridaSparseMatrixCollection.ACMTrans.Math.Softw.
38,1(2011),1:1–1:25.DOI:https://doi.org/10.1145/2049662.2049663
[39] GunduzVehbiDemirciandCevdetAykanat.2020.Cartesianpartitioningmodelsfor2Dand3DparallelSpGEMMal-
gorithms.IEEETrans.ParallelDistrib.Syst.31,12(2020),2763–2775.DOI:https://doi.org/10.1109/TPDS.2020.3000708
[40] GunduzVehbiDemirciandCevdetAykanat.2020.Scalingsparsematrix-matrixmultiplicationintheAccumulo
database.Distrib.ParallelDatab.38,1(2020),31–62.DOI:https://doi.org/10.1007/s10619-019-07257-y
[41] JulienDemouth.2012.Sparsematrix-matrixmultiplicationontheGPU.InGPUTechnologyConference.
[42] MehmetDeveci,ErikG.Boman,KarenD.Devine,andSivasankaranRajamanickam.2016.Parallelgraphcoloringfor
manycorearchitectures.InIEEEInternationalParallelandDistributedProcessingSymposium.IEEEComputerSociety,
892–901.DOI:https://doi.org/10.1109/IPDPS.2016.54
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:32 J.Gaoetal.
[43] MehmetDeveci,SimonD.Hammond,MichaelM.Wolf,andSivasankaranRajamanickam.2018.Sparsematrix-matrix
multiplicationonmultilevelmemoryarchitectures:Algorithmsandexperiments.CoRRabs/1804.00695(2018).
[44] Mehmet Deveci, Kamer Kaya, and Ümit V. Çatalyürek. 2013. Hypergraph sparsification and its application to
partitioning.In42ndInternationalConferenceonParallelProcessing.IEEEComputerSociety,200–209.DOI:https:
//doi.org/10.1109/ICPP.2013.29
[45] MehmetDeveci,ChristianTrott,andSivasankaranRajamanickam.2017.Performance-portablesparsematrix-matrix
multiplicationformany-corearchitectures.InIEEEInternationalParallelandDistributedProcessingSymposiumWork-
shops.693–702.DOI:https://doi.org/10.1109/IPDPSW.2017.8
[46] MehmetDeveci,ChristianTrott,andSivasankaranRajamanickam.2018.Multi-threadedsparsematrix-matrixmul-
tiplicationformany-coreandGPUarchitectures.CoRRabs/1801.03065(2018).
[47] H. Carter Edwards, Christian R. Trott, and Daniel Sunderland. 2014. Kokkos: Enabling manycore performance
portability through polymorphic memory access patterns. J. Parallel Distrib. Comput. 74, 12 (2014), 3202–3216.
DOI:https://doi.org/10.1016/j.jpdc.2014.07.003
[48] JamesJ.ElliottandChristopherM.Siefert.2018.Lowthread-countGustavson:Amultithreadedalgorithmforsparse
matrix-matrixmultiplicationusingperfecthashing.InIEEE/ACM9thWorkshoponLatestAdvancesinScalableAlgo-
rithmsforLarge-ScaleSystems(scalA).57–64.DOI:https://doi.org/10.1109/ScalA.2018.00011
[49] RobertD.Falgout.2006.Anintroductiontoalgebraicmultigrid.Comput.Sci.Eng.8,6(2006),24–33.DOI:https:
//doi.org/10.1109/MCSE.2006.105
[50] SalvatoreFilippone,ValeriaCardellini,DavideBarbieri,andAlessandroFanfarillo.2017.Sparsematrix-vectormul-
tiplicationonGPGPUs.ACMTrans.Math.Softw.43,4(2017),30:1–30:49.DOI:https://doi.org/10.1145/3017994
[51] VijayGadepally,JakeBolewski,DanHook,DylanHutchison,BenjaminA.Miller,andJeremyKepner.2015.Graphulo:
LinearalgebragraphkernelsforNoSQLdatabases.CoRRabs/1508.07372(2015).
[52] JohnR.Gilbert,CleveMoler,andRobertSchreiber.1992.SparsematricesinMATLAB:Designandimplementation.
SIAMJ.MatrixAnal.Appl.13,1(1992),333–356.DOI:https://doi.org/10.1137/0613024
[53] JohnR.Gilbert,StevenP.Reinhardt,andViralB.Shah.2006.High-performancegraphalgorithmsfromparallel
sparsematrices.In8thInternationalWorkshoponAppliedParallelComputing.StateoftheArtinScientificComputing
(LectureNotesinComputerScience,Vol.4699).Springer,260–269.DOI:https://doi.org/10.1007/978-3-540-75755-9_32
[54] JohnR.Gilbert,StevenP.Reinhardt,andViralB.Shah.2008.Aunifiedframeworkfornumericalandcombinatorial
computing.Comput.Sci.Eng.10,2(2008),20–25.DOI:https://doi.org/10.1109/MCSE.2008.45
[55] AshishGondimalla,NoahChesnut,MithunaThottethodi,andT.N.Vijaykumar.2019.SparTen:Asparsetensoraccel-
eratorforconvolutionalneuralnetworks.In52ndAnnualIEEE/ACMInternationalSymposiumonMicroarchitecture.
ACM,151–165.DOI:https://doi.org/10.1145/3352460.3358291
[56] Graphegon.2021.Pygraphblas.Retrievedfromhttps://github.com/Graphegon/pygraphblas.
[57] FelixGremse,AndreasHöfter,LarsOleSchwen,FabianKiessling,andUweNaumann.2015.GPU-acceleratedsparse
matrix-matrixmultiplicationbyiterativerowmerging.SIAMJ.Sci.Comput.37,1(2015).DOI:https://doi.org/10.1137/
130948811
[58] ZhixiangGu,JoseMoreira,DavidEdelsohn,andArifulAzad.2020.Bandwidthoptimizedparallelalgorithmsfor
sparsematrix-matrixmultiplicationusingpropagationblocking.In32ndACMSymposiumonParallelisminAlgo-
rithmsandArchitectures.ACM,293–303.DOI:https://doi.org/10.1145/3350755.3400216
[59] GiuliaGuidi,MarquitaEllis,DanielRokhsar,KatherineA.Yelick,andAydinBuluç.2021.BELLA:Berkeleyefficient
long-readtolong-readalignerandoverlapper.InSIAMConferenceonAppliedandComputationalDiscreteAlgorithms.
SIAM,123–134.DOI:https://doi.org/10.1137/1.9781611976830.12
[60] GiuliaGuidi,OguzSelvitopi,MarquitaEllis,LeonidOliker,KatherineA.Yelick,andAydinBuluç.2021.Parallelstring
graphconstructionandtransitivereductionfordenovogenomeassembly.In35thIEEEInternationalParalleland
DistributedProcessingSymposium.IEEE,517–526.DOI:https://doi.org/10.1109/IPDPS49936.2021.00060
[61] FredG.Gustavson.1978.Twofastalgorithmsforsparsematrices:Multiplicationandpermutedtransposition.ACM
Trans.Math.Softw.4,3(1978),250–269.DOI:https://doi.org/10.1145/355791.355796
[62] PouyaHaghi,TongGeng,AnqiGuo,TianqiWang,andMartinC.Herbordt.2020.FP-AMG:FPGA-basedaccelera-
tionframeworkforalgebraicmultigridsolvers.In28thIEEEAnnualInternationalSymposiumonField-Programmable
CustomComputingMachines.IEEE,148–156.DOI:https://doi.org/10.1109/FCCM48280.2020.00028
[63] SongHan,XingyuLiu,HuiziMao,JingPu,ArdavanPedram,MarkA.Horowitz,andWilliamJ.Dally.2016.EIE:
Efficientinferenceengineoncompresseddeepneuralnetwork.In43rdACM/IEEEAnnualInternationalSymposium
onComputerArchitecture.IEEEComputerSociety,243–254.DOI:https://doi.org/10.1109/ISCA.2016.30
[64] KartikHegde,HadiAsghariMoghaddam,MichaelPellauer,NealClaytonCrago,AamerJaleel,EdgarSolomonik,
JoelS.Emer,andChristopherW.Fletcher.2019.ExTensor:Anacceleratorforsparsetensoralgebra.In52ndAn-
nualIEEE/ACMInternationalSymposiumonMicroarchitecture.ACM,319–333.DOI:https://doi.org/10.1145/3352460.
3358275
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:33
[65] MichaelA.Heroux,RoscoeA.Bartlett,VickiE.Howle,RobertJ.Hoekstra,JonathanJ.Hu,TamaraG.Kolda,RichardB.
Lehoucq,KevinR.Long,RogerP.Pawlowski,EricT.Phipps,AndrewG.Salinger,HeidiThornquist,RayS.Tuminaro,
JamesM.Willenbring,AlanB.Williams,andKendallS.Stanley.2005.AnoverviewoftheTrilinosProject.ACMTrans.
Math.Softw.31,3(2005),397–423.DOI:https://doi.org/10.1145/1089014.1089021
[66] MdTaufiqueHussain,OguzSelvitopi,AydinBuluç,andArifulAzad.2021.Communication-avoidingandmemory-
constrainedsparsematrix-matrixmultiplicationatextremescale.In35thIEEEInternationalParallelandDistributed
ProcessingSymposium.IEEE,90–100.DOI:https://doi.org/10.1109/IPDPS49936.2021.00018
[67] DylanHutchison,JeremyKepner,VijayGadepally,andAdamFuchs.2015.Graphuloimplementationofserver-side
sparsematrixmultiplyintheAccumulodatabase.InIEEEHighPerformanceExtremeComputingConference.IEEE,
1–7.DOI:https://doi.org/10.1109/HPEC.2015.7322448
[68] Intel.2021.IntelMathKernelLibrary.Retrievedfromhttps://software.intel.com/en-us/mkl.
[69] FumiyaIshiguro,TakahiroKatagiri,SatoshiOhshima,andToruNagai.2020.Performanceevaluationofaccurate
matrix-matrixmultiplicationonGPUusingsparsematrixmultiplications.In8thInternationalSymposiumonCom-
putingandNetworkingWorkshops.IEEE,178–184.DOI:https://doi.org/10.1109/CANDARW51189.2020.00044
[70] ErnestJamro,TomaszPabis,PawelRussek,andKazimierzWiatr.2014.ThealgorithmsforFPGAimplementationof
sparsematricesmultiplication.Comput.Inform.33,3(2014),667–684.
[71] DejiangJinandSotiriosG.Ziavras.2004.Asuper-programmingtechniqueforlargesparsematrixmultiplicationon
PCclusters.IEICETrans.Inf.Syst.87-D,7(2004),1774–1781.
[72] KonstantinosKanellopoulos,NanditaVijaykumar,ChristinaGiannoula,RoknoddinAzizi,SkandaKoppula,Nika
Mansouri-Ghiasi,TahaShahroodi,JuanGómez-Luna,andOnurMutlu.2019.SMASH:Co-designingsoftwarecom-
pressionandhardware-acceleratedindexingforefficientsparsematrixoperations.In52ndAnnualIEEE/ACMInter-
nationalSymposiumonMicroarchitecture.ACM,600–614.DOI:https://doi.org/10.1145/3352460.3358286
[73] HaimKaplan,MichaSharir,andEladVerbin.2006.Coloredintersectionsearchingviasparserectangularmatrixmul-
tiplication.In22ndACMSymposiumonComputationalGeometry.ACM,52–60.DOI:https://doi.org/10.1145/1137856.
1137866
[74] BarbaraAnnKitchenham.2004.ProceduresforPerformingSystematicReviews.TechnicalReport.KeeleUniversity,
DepartmentofComputerScience,KeeleUniversity,Keele,UK.
[75] SüreyyaEmreKurt,VineethThumma,ChangwanHong,AravindSukumaran-Rajam,andP.Sadayappan.2017.Char-
acterizationofdatamovementrequirementsforsparsematrixcomputationsonGPUs.In24thIEEEInternationalCon-
ferenceonHighPerformanceComputing.IEEEComputerSociety,283–293.DOI:https://doi.org/10.1109/HiPC.2017.
00040
[76] RalfLämmel.2008.Google’sMapReduceprogrammingmodel-revisited.Sci.Comput.Program.70,1(2008),1–30.
DOI:https://doi.org/10.1016/j.scico.2007.07.001
[77] JeongmyungLee,SeokwonKang,YongseungYu,Yong-YeonJo,Sang-WookKim,andYongjunPark.2020.Optimiza-
tionofGPU-basedsparsematrixmultiplicationforlargesparsenetworks.In36thIEEEInternationalConferenceon
DataEngineering.IEEE,925–936.DOI:https://doi.org/10.1109/ICDE48307.2020.00085
[78] JiayuLi,FugangWang,TakuyaAraki,andJudyQiu.2019.Generalizedsparsematrix-matrixmultiplicationforvector
enginesandgraphapplications.InIEEE/ACMWorkshoponMemoryCentricHighPerformanceComputing.IEEE,33–
42.DOI:https://doi.org/10.1109/MCHPC49590.2019.00012
[79] ColinYuLin,NgaiWong,andHaydenKwok-HaySo.2013.Designspaceexplorationforsparsematrix-matrixmul-
tiplicationonFPGAs.Int.J.Circ.Theor.Appl.41,2(2013),205–219.DOI:https://doi.org/10.1002/cta.796
[80] JunhongLiu,XinHe,WeifengLiu,andGuangmingTan.2019.Register-awareoptimizationsforparallelsparsematrix-
matrixmultiplication.Int.J.ParallelProgram.47,3(2019),403–417.DOI:https://doi.org/10.1007/s10766-018-0604-8
[81] JiawenLiu,JieRen,RobertoGioiosa,DongLi,andJiajiaLi.2021.Sparta:High-performance,element-wisesparse
tensorcontractiononheterogeneousmemory.In26thACMSIGPLANSymposiumonPrinciplesandPracticeofParallel
Programming.ACM,318–333.DOI:https://doi.org/10.1145/3437801.3441581
[82] WeifengLiuandBrianVinter.2014.AnefficientGPUgeneralsparsematrix-matrixmultiplicationforirregulardata.
InIEEE28thInternationalParallelandDistributedProcessingSymposium.IEEEComputerSociety,370–381.DOI:https:
//doi.org/10.1109/IPDPS.2014.47
[83] WeifengLiuandBrianVinter.2015.Aframeworkforgeneralsparsematrix-matrixmultiplicationonGPUsand
heterogeneousprocessors.J.ParallelDistrib.Comput.85(2015),47–61.DOI:https://doi.org/10.1016/j.jpdc.2015.06.010
[84] KiranKumarMatam,SivaRamaKrishnaBharadwajIndarapu,andKishoreKothapalli.2012.Sparsematrix-matrix
multiplicationonmodernarchitectures.In19thInternationalConferenceonHighPerformanceComputing.IEEECom-
puterSociety,1–10.DOI:https://doi.org/10.1109/HiPC.2012.6507483
[85] Duane Merrill and Andrew S. Grimshaw. 2011. High performance and scalable radix sorting: A case study of
implementing dynamic parallelism for GPU computing. Parallel Process. Lett. 21, 2 (2011), 245–272. DOI:https:
//doi.org/10.1142/S0129626411000187
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:34 J.Gaoetal.
[86] Yusuke Nagasaka, Satoshi Matsuoka, Ariful Azad, and Aydin Buluç. 2019. Performance optimization, modeling
andanalysisofsparsematrix-matrixproductsonmulti-coreandmany-coreprocessors.ParallelComput.90(2019).
DOI:https://doi.org/10.1016/j.parco.2019.102545
[87] YusukeNagasaka,AkiraNukada,andSatoshiMatsuoka.2017.High-performanceandmemory-savingsparsegeneral
matrix-matrixmultiplicationforNVIDIAPascalGPU.In46thInternationalConferenceonParallelProcessing.IEEE
ComputerSociety,101–110.DOI:https://doi.org/10.1109/ICPP.2017.19
[88] MaximNaumov,M.Arsaev,PatriceCastonguay,JonathanM.Cohen,JulienDemouth,JoeEaton,SimonK.Layton,
N.Markovskiy,IstvánZ.Reguly,NikolaiSakharnykh,V.Sellappan,andRobertStrzodka.2015.AmgX:Alibrary
forGPUacceleratedalgebraicmultigridandpreconditionediterativemethods.SIAMJ.Sci.Comput.37,5(2015),
S602–S626.DOI:https://doi.org/10.1137/140980260
[89] YuyaoNiu,ZhengyangLu,HaonanJi,ShuhuiSong,ZhouJin,andWeifengLiu.2022.TileSpGEMM:Atiledalgorithm
forparallelsparsegeneralmatrix-matrixmultiplicationonGPUs.In27thACMSIGPLANSymposiumonPrinciples
andPracticeofParallelProgramming.ACM,90–106.DOI:https://doi.org/10.1145/3503221.3508431
[90] NVIDIA.2021.NvidiacuSPARSElibrary.Retrievedfromhttps://developer.nvidia.com/cusparse.
[91] SubhankarPal,JonathanBeaumont,Dong-HyeonPark,AporvaAmarnath,SiyingFeng,ChaitaliChakrabarti,Hun-
SeokKim,DavidT.Blaauw,TrevorN.Mudge,andRonaldG.Dreslinski.2018.OuterSPACE:Anouterproductbased
sparsematrixmultiplicationaccelerator.InIEEEInternationalSymposiumonHighPerformanceComputerArchitecture.
IEEEComputerSociety,724–736.DOI:https://doi.org/10.1109/HPCA.2018.00067
[92] Mathias Parger, Martin Winter, Daniel Mlakar, and Markus Steinberger. 2020. spECK: Accelerating GPU sparse
matrix-matrixmultiplicationthroughlightweightanalysis.In25thACMSIGPLANSymposiumonPrinciplesandPrac-
ticeofParallelProgramming.ACM,362–375.DOI:https://doi.org/10.1145/3332466.3374521
[93] SehunPark,Jae-JoonKim,andJaehaKung.2022.AutoRelax:HW-SWco-optimizationforefficientSpGEMMop-
erations with automated relaxation in deep learning. IEEE Trans. Emerg. Top. Comput. 10, 3 (2022), 1428–1442.
DOI:https://doi.org/10.1109/TETC.2021.3089848
[94] Md.MostofaAliPatwary,NadathurRajagopalanSatish,NarayananSundaram,JongsooPark,MichaelJ.Anderson,
SatyaGautamVadlamudi,DipankarDas,SergeyG.Pudov,VadimO.Pirogov,andPradeepDubey.2015.Parallel
efficientsparsematrix-matrixmultiplicationonmulticoreplatforms.In30thISCInternationalConferenceonHigh
PerformanceComputing(LectureNotesinComputerScience,Vol.9137).Springer,48–57.DOI:https://doi.org/10.1007/
978-3-319-20119-1_4
[95] LillianPentecost,MarcoDonato,BrandonReagen,UditGupta,SimingMa,Gu-YeonWei,andDavidBrooks.2019.
MaxNVM:MaximizingDNNstoragedensityandinferenceefficiencywithsparseencodinganderrormitigation.In
52ndAnnualIEEE/ACMInternationalSymposiumonMicroarchitecture.ACM,769–781.DOI:https://doi.org/10.1145/
3352460.3358258
[96] ChuckPheatt.2008.Intel®threadingbuildingblocks.J.Comput.Sci.Coll.23,4(2008),298–298.
[97] EricQin,GeonhwaJeong,WilliamWon,Sheng-ChunKao,HyoukjunKwon,SudarshanSrinivasan,DipankarDas,
GordonEuhyunMoon,SivasankaranRajamanickam,andTusharKrishna.2021.Extendingsparsetensoraccelerators
tosupportmultiplecompressionformats.In35thIEEEInternationalParallelandDistributedProcessingSymposium.
IEEE,1014–1024.DOI:https://doi.org/10.1109/IPDPS49936.2021.00110
[98] EricQin,AnandaSamajdar,HyoukjunKwon,VineetNadella,SudarshanSrinivasan,DipankarDas,BharatKaul,
andTusharKrishna.2020.SIGMA:AsparseandirregularGEMMacceleratorwithflexibleinterconnectsforDNN
training.InIEEEInternationalSymposiumonHighPerformanceComputerArchitecture.IEEE,58–70.DOI:https://doi.
org/10.1109/HPCA47549.2020.00015
[99] SivasankaranRajamanickam,SeherAcer,LucBerger-Vergiat,VinhQ.Dang,NathanD.Ellingwood,EvanHarvey,
BrianKelley,ChristianR.Trott,JeremiahJ.Wilke,andIchitaroYamazaki.2021.Kokkoskernels:Performanceportable
sparse/denselinearalgebraandgraphkernels.CoRRabs/2103.11991(2021).
[100] AkshayKrishnaRamanathan,SrivatsaSrinivasaRangachar,HariramThirucheraiGovindarajan,Je-MinHung,Chun-
YingLee,Cheng-XinXue,Sheng-PoHuang,Fu-KuoHsueh,Chang-HongShen,Jia-MinShieh,Wen-KuanYeh,Mon-
ShuHo,JackSampson,Meng-FanChang,andVijaykrishnanNarayanan.2021.CiM3D:Comparator-in-memoryde-
signsusingmonolithic3-Dtechnologyforacceleratingdata-intensiveapplications.IEEEJ.Explor.Solid-StateCom-
putat.Dev.Circ.7,1(2021),79–87.DOI:https://doi.org/10.1109/JXCDC.2021.3087745
[101] AkshayKrishnaRamanathan,SrivatsaSrinivasaRangachar,Je-MinHung,Chun-YingLee,Cheng-XinXue,Sheng-Po
Huang,Fu-KuoHsueh,Chang-HongShen,Jia-MinShieh,Wen-KuanYeh,Mon-ShuHo,HariramThirucheraiGovin-
darajan,JackSampson,Meng-FanChang,andVijaykrishnanNarayanan.2020.Monolithic3D+-ICbasedmassively
parallelcompute-in-memorymacroforacceleratingdatabaseandmachinelearningprimitives.InIEEEInternational
ElectronDevicesMeeting(IEDM).28.5.1–28.5.4.DOI:https://doi.org/10.1109/IEDM13553.2020.9372111
[102] MajidRasouli,RobertM.Kirby,andHariSundar.2021.Acompressed,divideandconqueralgorithmforscalable
distributedmatrix-matrixmultiplication.InInternationalConferenceonHighPerformanceComputinginAsia-Pacific
Region.ACM,110–119.DOI:https://doi.org/10.1145/3432261.3432271
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

ASystematicSurveyofGeneralSparseMatrix-matrixMultiplication 244:35
[103] ThomasB.Rolinger,ChristopherD.Krieger,andAlanSussman.2021.Optimizingmemory-computecolocationfor
irregularapplicationsonamigratorythreadarchitecture.In35thIEEEInternationalParallelandDistributedProcessing
Symposium.IEEE,58–67.DOI:https://doi.org/10.1109/IPDPS49936.2021.00015
[104] EmanuelH.RubenssonandEliasRudberg.2014.Chunksandtasks:Aprogrammingmodelforparallelizationof
dynamicalgorithms.ParallelComput.40,7(2014),328–343.DOI:https://doi.org/10.1016/j.parco.2013.09.006
[105] EmanuelH.RubenssonandEliasRudberg.2016.Locality-awareparallelblock-sparsematrix-matrixmultiplication
usingthechunksandtasksprogrammingmodel.ParallelComput.57(2016),87–106.DOI:https://doi.org/10.1016/j.
parco.2016.06.005
[106] MarzioSala,WilliamF.Spotz,andMichaelA.Heroux.2008.PyTrilinos: High-performancedistributed-memory
solversforpython.ACMTrans.Math.Softw.34,2(2008),7:1–7:33.DOI:https://doi.org/10.1145/1326548.1326549
[107] OguzSelvitopi,GunduzVehbiDemirci,AtaTurk,andCevdetAykanat.2019.Locality-awareandload-balancedstatic
taskschedulingforMapReduce.Fut.Gen.Comput.Syst.90(2019),49–61.DOI:https://doi.org/10.1016/j.future.2018.
06.035
[108] OguzSelvitopi,SaliyaEkanayake,GiuliaGuidi,GeorgiosA.Pavlopoulos,ArifulAzad,andAydinBuluç.2020.Dis-
tributedmany-to-manyproteinsequencealignmentusingsparsematrices.InInternationalConferenceforHighPer-
formanceComputing,Networking,StorageandAnalysis.IEEE/ACM,75.DOI:https://doi.org/10.1109/SC41405.2020.
00079
[109] OguzSelvitopi,MdTaufiqueHussain,ArifulAzad,andAydinBuluç.2020.OptimizinghighperformanceMarkov
clusteringforpre-exascalearchitectures.InIEEEInternationalParallelandDistributedProcessingSymposium.IEEE,
116–126.DOI:https://doi.org/10.1109/IPDPS47924.2020.00022
[110] KaustubhShivdikar.2021.SMASH:Sparsematrixatomicscratchpadhashing.CoRRabs/2105.14156(2021).
[111] JakobSiegel,OresteVilla,SriramKrishnamoorthy,AntoninoTumeo,andXiaomingLi.2010.Efficientsparsematrix-
matrix multiplication on heterogeneous high performance systems. In IEEE International Conference on Cluster
ComputingWorkshopsandPosters(CLUSTERWORKSHOPS).1–8.DOI:https://doi.org/10.1109/CLUSTERWKSP.2010.
5613109
[112] EdgarSolomonik,DevinMatthews,JeffR.Hammond,JohnF.Stanton,andJamesDemmel.2014.Amassivelyparallel
tensorcontractionframeworkforcoupled-clustercomputations.J.ParallelDistrib.Comput.74,12(2014),3176–3190.
DOI:https://doi.org/10.1016/j.jpdc.2014.06.002
[113] MohammadrezaSoltaniyeh,RichardP.Martin,andSantoshNagarakatte.2020.SynergisticCPU-FPGAacceleration
ofsparselinearalgebra.CoRRabs/2004.13907(2020).
[114] SriseshanSrikanth,AnirudhJain,JosephM.Lennon,ThomasM.Conte,ErikDeBenedictis,andJeanineE.Cook.
2020.MetaStrider:Architecturesforscalablememory-centricreductionofsparsedatastreams.ACMTrans.Archit.
CodeOptim.16,4(2020),35:1–35:26.DOI:https://doi.org/10.1145/3355396
[115] NitishKumarSrivastava,HanchenJin,JieLiu,DavidH.Albonesi,andZhiruZhang.2020.MatRaptor:Asparse-sparse
matrixmultiplicationacceleratorbasedonrow-wiseproduct.In53rdAnnualIEEE/ACMInternationalSymposiumon
Microarchitecture.IEEE,766–780.DOI:https://doi.org/10.1109/MICRO50266.2020.00068
[116] ManuelThen,MoritzKaufmann,FernandoChirigati,Tuan-AnhHoang-Vu,KienPham,AlfonsKemper,Thomas
Neumann,andHuyT.Vo.2014.Themorethemerrier:Efficientmulti-sourcegraphtraversal.Proc.VLDBEndow.8,
4(2014),449–460.DOI:https://doi.org/10.14778/2735496.2735507
[117] TheTrilinosProjectTeam.2020.TheTrilinosHomePage.Retrievedfromhttps://trilinos.github.io.
[118] Robert A. van de Geijn and Jerrell Watts. 1997. SUMMA: Scalable universal matrix multiplication algorithm.
Concurr.Pract.Exp.9,4(1997),255–274.DOI:https://doi.org/10.1002/(SICI)1096-9128(199704)9:4<255::AID-CPE250>
3.0.CO;2-2
[119] VirginiaVassilevska,RyanWilliams,andRaphaelYuster.2006.Findingheaviesth-subgraphsinrealweightedgraphs,
withapplications.CoRRabs/cs/0609009(2006).
[120] XiaoyunWang,ZhongyiLin,CarlYang,andJohnD.Owens.2019.AcceleratingDNNinferencewithGraphBLAS
andtheGPU.InIEEEHighPerformanceExtremeComputingConference.IEEE,1–6.DOI:https://doi.org/10.1109/HPEC.
2019.8916498
[121] ValéryWeber,TeodoroLaino,AlexanderPozdneev,IrinaFedulova,andAlessandroCurioni.2015.Semiempirical
moleculardynamics(SEMD)I:Midpoint-basedparallelsparsematrix-matrixmultiplicationalgorithmformatrices
withdecay.J.Chem.Theor.Computat.11,7(2015),3145–3152.DOI:https://doi.org/10.1021/acs.jctc.5b00382
[122] Martin Winter, Daniel Mlakar, Rhaleb Zayer, Hans-Peter Seidel, and Markus Steinberger. 2019. Adaptive sparse
matrix-matrixmultiplicationontheGPU.In24thACMSIGPLANSymposiumonPrinciplesandPracticeofParallel
Programming.ACM,68–81.DOI:https://doi.org/10.1145/3293883.3295701
[123] MichaelM.Wolf,JonathanW.Berry,andDylanT.Stark.2015.Atask-basedlinearalgebrabuildingblocksapproach
forscalablegraphanalytics.InIEEEHighPerformanceExtremeComputingConference.IEEE,1–6.DOI:https://doi.
org/10.1109/HPEC.2015.7322450
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.

244:36 J.Gaoetal.
[124] MichaelM.Wolf,MehmetDeveci,JonathanW.Berry,SimonD.Hammond,andSivasankaranRajamanickam.2017.
Fastlinearalgebra-basedtrianglecountingwithKokkosKernels.InIEEEHighPerformanceExtremeComputingCon-
ference.IEEE,1–7.DOI:https://doi.org/10.1109/HPEC.2017.8091043
[125] YangXia,PengJiang,GaganAgrawal,andRajivRamnath.2021.ScalingsparsematrixmultiplicationonCPU-GPU
nodes.In35thIEEEInternationalParallelandDistributedProcessingSymposium.IEEE,392–401.DOI:https://doi.org/
10.1109/IPDPS49936.2021.00047
[126] JiamingXieandYunLiang.2019.SPART:OptimizingCNNsbyutilizingbothsparsityofweightsandfeaturemaps.
In13thInternationalSymposiumonAdvancedParallelProcessingTechnologies(LectureNotesinComputerScience,
Vol.11719).Springer,71–85.DOI:https://doi.org/10.1007/978-3-030-29611-7_6
[127] AbdurrahmanYasar,SivasankaranRajamanickam,MichaelM.Wolf,JonathanW.Berry,andÜmitV.Çatalyürek.2018.
FasttrianglecountingusingCilk.InIEEEHighPerformanceExtremeComputingConference.IEEE,1–7.DOI:https:
//doi.org/10.1109/HPEC.2018.8547563
[128] RaphaelYusterandUriZwick.2005.Fastsparsematrixmultiplication.ACMTrans.Algor.1,1(2005),2–13.DOI:https:
//doi.org/10.1145/1077464.1077466
[129] Orestis Zachariadis, Nitin Satpute, Juan Gómez-Luna, and Joaquín Olivares. 2020. Accelerating sparse matrix-
matrixmultiplicationwithGPUtensorcores.Comput.Electr.Eng.88(2020),106848.DOI:https://doi.org/10.1016/
j.compeleceng.2020.106848
[130] FengZhang,WeifengLiu,NingxuanFeng,JidongZhai,andXiaoyongDu.2019.Performanceevaluationandanalysis
ofsparsematrixandgraphkernelsonheterogeneousprocessors.CCFTrans.HighPerform.Comput.1,2(2019),131–
143.DOI:https://doi.org/10.1007/s42514-019-00008-6
[131] GuoweiZhang,NithyaAttaluri,JoelS.Emer,andDanielSánchez.2021.Gamma:LeveragingGustavson’salgorithm
toacceleratesparsematrixmultiplication.In26thACMInternationalConferenceonArchitecturalSupportforProgram-
mingLanguagesandOperatingSystems.ACM,687–701.DOI:https://doi.org/10.1145/3445814.3446702
[132] YudongZhang,LenanWu,GengWei,andShuihuaWang.2011.Anovelalgorithmforallpairsshortestpathproblem
basedonmatrixmultiplicationandpulsecoupledneuralnetwork.Digit.Sig.Process.21,4(2011),517–521.DOI:https:
//doi.org/10.1016/j.dsp.2011.02.004
[133] ZhekaiZhang,HanruiWang,SongHan,andWilliamJ.Dally.2020.SpArch:Efficientarchitectureforsparsema-
trix multiplication. In IEEE International Symposium on High Performance Computer Architecture. IEEE, 261–274.
DOI:https://doi.org/10.1109/HPCA47549.2020.00030
Received2March2022;revised12September2022;accepted28October2022
ACMComputingSurveys,Vol.55,No.12,Article244.Publicationdate:March2023.
