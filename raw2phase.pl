#! /usr/bin/perl
# @author Shizuo KAJI
# @date 3 Oct. 2017
# @copyright The MIT License

use strict;
use warnings;

# prohibited patterns in SMILES
my $prohibited = '[^cCHONSFClBrI\+\-\d\=\#\%\(\)\[\]]';
# ignore PT entries and focus on Phases entries
my $ignorePT = 1;

# print usage
if(@ARGV < 2){
	print "Usage: $0 smiles raw truth > output_file \n";
	exit;
}

# read the input file
my $smifile = shift;
my $rawfile = shift;
open(my $smifh, "<", $smifile) or die "Cannot open $smifile: $!";
my @smi = readline $smifh;
close $smifh;
open(my $rawfh, "<", $rawfile) or die "Cannot open $rawfile: $!";
my @raw = readline $rawfh;
close $rawfh;

my @truth;
if(@ARGV == 3) {
	my $truthfile = shift;
	open(my $truthfh, "<", $truthfile) or die "Cannot open $truthfile: $!";
	@truth = readline $truthfh;
	close $truthfh;
}else{
	for (my $i = 0; $i <= $#smi; $i++){
		$truth[$i] = 'true';
	}
}

# parse the phase string
## @arr is global
our %arr;
sub analysePhase {
	my ($pline) = @_;
	my $updated = 0;
	$pline =~ s/\s/\t/g;
	$pline =~ s/\s0(\sX\s0)+\s/ /g; # replace 0 X 0 X ... X 0 to X (monotropic)
	$pline =~ s/\s0(\s0)+\s/ /g; # remove 0 0 ... 0 
	$pline =~ s/\sD\s/ Y /g; # D => Y
	$pline =~ s/\sDh[od]*\s/ D /g; # Dh, Dho, Dhp => D
	# melting
	unless($arr{"Melting"}>-300){
#		print "\n$pline\n$arr{'Melting'}\n";
		if ($pline =~ /Cr\s+(-?\d+(\.\d+)?)\s+/){
			$arr{"Melting_type"} = 1;
			$arr{"Melting"} = $1;
			$updated = 1;
		}elsif ($pline =~ /Cr1\s+(-?\d+(\.\d+)?)\s+/){
			$arr{"Melting_type"} = 3;  ## second Cr: Cr2 -> Cr1 -> X
			$arr{"Melting"} = $1;
			$updated = 1;
		}elsif ($pline =~ /Tg\s+(-?\d+(\.\d+)?)\s+/){
			$arr{"Melting_type"} = 2;  ## glass
			$arr{"Melting"} = $1;
			$updated = 1;
		}else{
			$arr{"Melting_type"} = 0;  # unknown melting
		}
	}
	# is phase
	unless($arr{"Clearing"}>-300){
		if ($pline =~ /\s+(-?\d+(\.\d+)?)\s+is/){
			$arr{"Clearing_type"} = 1;
			$arr{"Clearing"} = $1;
			$updated = 1;
		}elsif ($pline =~ /\s+(-?\d+(\.\d+)?)\s+ex/){
			$arr{"Clearing_type"} = 2;
			$arr{"Clearing"} = $1;
			$updated = 1;
		}elsif ($pline =~ /\s+(-?\d+(\.\d+)?)\s+chg/){
			$arr{"Clearing_type"} = 3;
			$arr{"Clearing"} = $1;
			$updated = 1;
		}elsif ($pline =~ /\s+(-?\d+(\.\d+)?)\s+dec/){
			$arr{"Clearing_type"} = 4;
			$arr{"Clearing"} = $1;
			$updated = 1;
		}elsif ($pline =~ /\sis/){
			$arr{"Clearing_type"} = 1;
		}elsif ($pline =~ /\sex/){
			$arr{"Clearing_type"} = 2;
		}elsif ($pline =~ /\schg/){
			$arr{"Clearing_type"} = 3;
		}else{
			$arr{"Clearing_type"} = 0;
		}
	}
#	print "$pline\n";
#	print "$arr{'Clearing'}\n";
	# phases:  type = 0 (none) 1 (normal) 2 (*)
	foreach my $ps ("B","C","A","S","N","D","X","Y"){
		# below
		unless($arr{$ps."m"}>-300){
			if ($pline =~ /\s+(-?\d+(\.\d+)?)\s+$ps\s/){
				$arr{$ps."type"} = 1;
				$arr{$ps."m"} = $1;
				$updated = 1;
			}elsif ($pline =~ /\s+(-?\d+(\.\d+)?)\s+$ps\*\s/){
				$arr{$ps."type"} = 2;
				$arr{$ps."m"} = $1;
				$updated = 1;
			}elsif ($pline =~ /\s$ps\s/){
				$arr{$ps."type"} = 1;
			}elsif ($pline =~ /\s$ps\*\s/){
				$arr{$ps."type"} = 2;
			}else{
				$arr{$ps."type"} = 0;
			}
		}
			# above
		unless($arr{$ps."p"}>-300){
			if ($pline =~ /.+\s$ps\s+(-?\d+(\.\d+)?)\s/){  # .+ forces last match
				$arr{$ps."type"} = 1;
				$arr{$ps."p"} = $1;
				$updated = 1;
			}elsif ($pline =~ /.+\s$ps\*\s+(-?\d+(\.\d+)?)\s/){ # .+ forces last match
				$arr{$ps."type"} = 2;
				$arr{$ps."p"} = $1;
				$updated = 1;
			}elsif ($pline =~ /\s$ps\s/){
				$arr{$ps."type"} = 1;
			}elsif ($pline =~ /\s$ps\*\s/){
				$arr{$ps."type"} = 2;
			}else{
				$arr{$ps."type"} = 0;
			}
		}
	}
	return $updated;
}

sub printline {
	my ($prev_line,$smiles,$phs) = @_;
	if($prev_line =~ /true\t/){
		$prev_line =~ s/true\t//;
		if($prev_line !~ /dup\t/){  ## duplicate
			$prev_line = $prev_line.",$smiles,$phs,$arr{'rac_en'},$arr{'Melting_type'},$arr{'Melting'},$arr{'Btype'},$arr{'Bm'},$arr{'Bp'},$arr{'Ctype'},$arr{'Cm'},$arr{'Cp'},$arr{'Atype'},$arr{'Am'},$arr{'Ap'},$arr{'Stype'},$arr{'Ntype'},$arr{'Nm'},$arr{'Np'},$arr{'Dtype'},$arr{'Dm'},$arr{'Dp'},$arr{'Xtype'},$arr{'Ytype'},$arr{'Clearing_type'},$arr{'Clearing'},$arr{'num_C'},$arr{'num_H'},$arr{'num_N'}";
			## pattern check flag
			if($smiles !~ /$prohibited/){
				print $prev_line.",0\n";
			}else{
				print $prev_line.",1\n";
			}
		}
	}
}

## start here
## print header
print "#SMILES ID,SMILES,Phases,rac_en,Melting_type,Melting,Btype,Bm,Bp,Ctype,Cm,Cp,Atype,Am,Ap,Stype,Ntype,Nm,Np,Dtype,Dm,Dp,Xtype,Ytype,Clearing_type,Clearing,num_C,num_H,num_N,prohibited\n";

my $prev_line="";
my $smiles = "";
my $new_smiles = "";
my $phs = "NEW";
my $dup = 0;   ## flag for duplicate entry
my $count = 0;
# ID,smiles,Phase
foreach my $line (@raw){
	chomp($line);
	if ($line =~ /^[C\.]omp\.\sID:\s\[(\d+)\]/){
		my $id = $1;
		$new_smiles = shift(@smi);
		chomp($new_smiles);
		$new_smiles =~ s/Smiles\s+([^,]+)/$1/;
		$new_smiles =~ s/\r//g;
		## terminate the previous entry
		if($new_smiles ne $smiles){
			if($dup != 0){
				$dup = 0;
				$prev_line = "dup\t".$prev_line;
			}
			$count++;
		}else{
			$dup = 1;
			$prev_line = "dup\t".$prev_line;
		}
		&printline($prev_line,$smiles,$phs);
#		if($count>1){ exit; }
#		print("\n\n-----------------\n\n");

		## begin a new entry
		%arr = ("rac_en" => 0);
		for my $ps ("Melting","Clearing","Bp","Cp","Ap","Sp","Np","Bm","Cm","Am","Sm","Nm","Dm","Dp","Xm","Xp","Ym","Yp"){
			$arr{$ps} = -300;
		}
		for my $ps ("num_C","num_N","num_H"){
			$arr{$ps} = 0;
		}
		$phs = "NEW";
		$smiles = $new_smiles;
		my $tr = shift(@truth);
		chomp($tr);
		$prev_line = $tr."\t".$smiles."\t".$id; 
	}elsif($line =~ /^(Phases:)\s(.+)/){
		my $newphs = $2;
		$newphs =~ s/\.r/Cr/g;
		$newphs =~ s/\s\./ C/g;
		$newphs =~ s/\,/ /g;
		my $updated = &analysePhase($newphs);
		if( $updated>0 || $phs =~ /NEW/){
			$phs = $newphs;
		}
	}elsif($line =~ /^(PT)\s(.+)/ && $ignorePT == 0){
		my $newphs = $2;
		$newphs =~ s/\.r/Cr/g;
		$newphs =~ s/\s\./ C/g;
		$newphs =~ s/\,/ /g;
		my $updated = &analysePhase($newphs);
		if( $updated>0 || $phs =~ /NEW/){
			$phs = $newphs;
		}
	}elsif($line =~ /^(Formula:)(.+)/){
		my $newformula = $2;
		$newformula =~ s/\./C/;
		if($newformula =~ /\sC(\d+)/){
			$arr{'num_C'} = $1;
		}
		if($newformula =~ /\sN(\d+)/){
			$arr{'num_N'} = $1;
		}
		if($newformula =~ /\sH(\d+)/){
			$arr{'num_H'} = $1;
		}
	}elsif($line =~ /Derivat\spure\senantiomer/){
		$arr{'rac_en'} = "pure_en";
	}elsif($line =~ /Derivat\senantiomer/){
		$arr{'rac_en'} = "en";
	}elsif($line =~ /Derivat\sracemate/){
		$arr{'rac_en'} = "rac";
	}
}
## terminate the last entry
&printline($prev_line,$smiles,$phs);
