print "Reading latexmkrc!\n";

$pdf_mode = 1;
$bibtex = 'biber %O %B';
$out_dir = 'build';
$aux_dir = 'build';

END {
    use File::Copy;
    copy("build/main.pdf", "main.pdf");
}