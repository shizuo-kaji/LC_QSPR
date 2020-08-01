#! /usr/bin/perl
# @author Shizuo KAJI
# @date 3 Oct. 2017
# @copyright The MIT License

use strict;
use warnings;
use Getopt::Long 'GetOptions';

# print usage
if(@ARGV < 1) {
         print "Usage: $0 csvfile > output_file \n";
		 exit;
}

# read the input file
my $csvfile = shift;
open(my $csvfh, "<", $csvfile) or die "Cannot open $csvfile: $!";
my @csv = readline $csvfh;
close $csvfh;

my $linenum = 1;
foreach my $pline (@csv) {
	if($linenum==1){
		$pline =~ s/^(.+?),//;  # remove the first entry
		print "ID,SMILES,Phases,rac_en,Melting_type,Melting,Bmtype,Bm,Bptype,Bp,Cmtype,Cm,Cptype,Cp,Amtype,Am,Aptype,Ap,Smtype,Sm,Sptype,Sp,Nmtype,Nm,Nptype,Np,Clearing_type,Clearing,prohibited,".$pline;
	}else{
		$pline =~ s/\"//g;
		$pline =~ s/\t/ /g;
		print $pline;
	}
	$linenum += 1
}
