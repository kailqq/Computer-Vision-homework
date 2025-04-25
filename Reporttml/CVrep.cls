\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{CVrep}[CV report template]

\LoadClass[a4paper,12pt]{article}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{array}
\geometry{left=3cm,right=3cm,top=3cm,bottom=3cm}
\RequirePackage{ctex}
\RequirePackage{arydshln} % 添加arydshln包以支持虚线
\RequirePackage{cite} % 添加引用包
\RequirePackage{minted}
\usemintedstyle{one-dark}
\setminted{
    frame=none,
    baselinestretch=1.2,
    fontsize=\footnotesize,
    linenos,
    bgcolor=gray!10
}

\newcommand{\reporttitle}[1]{\def\@reporttitle{#1}}
\newcommand{\studentname}[1]{\def\@studentname{#1}}
\newcommand{\studentid}[1]{\def\@studentid{#1}}
\newcommand{\email}[1]{\def\@studentemail{#1}}
\newcommand{\phone}[1]{\def\@studentphone{#1}}
\newcommand{\advisor}[1]{\def\@advisor{#1}}
\renewcommand{\maketitle}{
    \begin{titlepage}
        \centering

        \vspace*{1cm}
        
        \includegraphics[width=0.6\textwidth]{./img/zju_logo.png}
        
        \vspace{2cm} 

        {\LARGE \textbf{计算机视觉(本科)作业报告}}
        
        \vspace{1cm}
        \begin{center}
            \zihao{4}
            \renewcommand{\arraystretch}{2}
            \begin{tabular}{rp{75mm}<{\centering\arraybackslash}}
            作业名称： & \@reporttitle \\ \cline{2-2}
            姓\hspace{1em}名： & \@studentname \\ \cline{2-2}
            学\hspace{1em}号： & \@studentid \\ \cline{2-2}
            电子邮箱： & \@studentemail \\ \cline{2-2}
            联系电话： & \@studentphone \\ \cline{2-2}
            指导老师： & \@advisor \\ \cline{2-2}
            \end{tabular}
        \end{center}
        \vfill
        \vspace{1cm}
        \number\year 年 \number\month 月 \number\day 日
    \end{titlepage}
}
\endinput