/*
This Software is distributed with the following X11 License,
sometimes also known as MIT license.
 
Copyright (c) 2010 Miguel A. Ballicora

 Permission is hereby granted, free of charge, to any person
 obtaining a copy of this software and associated documentation
 files (the "Software"), to deal in the Software without
 restriction, including without limitation the rights to use,
 copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following
 conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 OTHER DEALINGS IN THE SOFTWARE.
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "gtb-probe.h"

/* local prototypes */
static void dtm_print (unsigned stm, int tb_available, unsigned info, unsigned pliestomate);
static void wdl_print (unsigned stm, int tb_available, unsigned info);

/* 	
|	paths to TBs, generally provided by the user 
|	Two styles are accepted: One path at a time, or
|	multiple paths separated by ';'
|	The example in this file uses both styles simultaneoulsy
*/
const char *path1 = "gtb/gtb4";
const char *path2 = "gtb/gtb3;gtb/gtb2";
const char *path3 = "gtb/gtb1";

int main (int argc, char *argv[])
{
	/*--------------------------------------*\
	|   Probing info to be provided
	\*--------------------------------------*/

	unsigned int  stm;		/* side to move */
	unsigned int  epsquare;	/* target square for an en passant capture */
	unsigned int  castling;	/* castling availability, 0 => no castles */
	unsigned int  ws[17];	/* list of squares for white */
	unsigned int  bs[17];	/* list of squares for black */
	unsigned char wp[17];	/* what white pieces are on those squares */
	unsigned char bp[17];	/* what black pieces are on those squares */

	/*--------------------------------------*\
	|   Probing info to be requested
	\*--------------------------------------*/

	char *initinfo;				/* NULL if verbosity=0, initialization info if verbosity=1*/
	int tb_available;			/* 0 => FALSE, 1 => TRUE */
	unsigned info = tb_UNKNOWN;	/* default, no tbvalue */
	unsigned pliestomate;	

	/*--------------------------------------*\
	|   Initialization info to be provided
	\*--------------------------------------*/

	int verbosity = 1;		/* initialization 0 = non-verbose, 1 = verbose */
	int	scheme = tb_CP4;	/* compression scheme to be used */
	const char ** paths;	/* paths where files will be searched */
	size_t cache_size = 32*1024*1024; /* 32 MiB in this example */
	
	/* 	wdl_fraction:
		fraction, over 128, that will be dedicated to wdl information. 
		In other words, 96 means 3/4 of the cache will be dedicated to 
		win-draw-loss info, and 1/4 dedicated to distance to mate 
		information. 
	*/
	int wdl_fraction = 96; 

	/*----------------------------------*\
	|	Return version of this demo
	\*----------------------------------*/

	#include "version.h"
	#include "progname.h"
	if (argc > 1 && 0==strcmp(argv[1],"-v")) {
		printf ("%s %s\n",PROGRAM_NAME,VERSION);
		return 0;
	}

	/*--------------------------------------*\
	|   Initialization:
	|   Include something like this at
	|   the beginning of the program.   
	\*--------------------------------------*/

	/* the number of paths that can be added is only limited by memory */
	paths = tbpaths_init();
if (NULL == paths) printf ("Error here... %d\n",__LINE__);
	paths = tbpaths_add (paths, path1);
if (NULL == paths) printf ("Error here... %d\n",__LINE__);
	paths = tbpaths_add (paths, path2);
if (NULL == paths) printf ("Error here... %d\n",__LINE__);
	paths = tbpaths_add (paths, path3);
if (NULL == paths) printf ("Error here... %d\n",__LINE__);

	/* init probing code, indexes, paths, etc. */
	initinfo = tb_init (verbosity, scheme, paths);

	/* init cache */
	tbcache_init(cache_size, wdl_fraction); 

	tbstats_reset();

	/* information to be output for the user, or to be saved in logs etc.*/
	if (initinfo != NULL)
		printf ("%s",initinfo);

	/*--------------------------------------*\
	|
	|   ASSIGNING POSITIONAL VALUES for
	|   one probing example
	|   
	\*--------------------------------------*/

#if 1

	/* needs 3-pc installed */
	/* FEN: 8/8/8/4k3/8/8/8/KR6 w - - 0 1 */

	stm      = tb_WHITE_TO_MOVE;/* 0 = white to move, 1 = black to move */
	epsquare = tb_NOSQUARE;		/* no ep available */
	castling = tb_NOCASTLE;		/* no castling available, otherwise combine all 
									the castling possibilities with '|', for instance
									white could castle both sides, black can't:	 
									castling = tb_WOO | tb_WOOO; 
									both could castle on the king side:	 
									castling = tb_WOO | tb_WOO;
									etc. 
								*/

	ws[0] = tb_A1;
	ws[1] = tb_B1;
	ws[2] = tb_NOSQUARE;		/* it marks the end of list */

	wp[0] = tb_KING;
	wp[1] = tb_ROOK;
	wp[2] = tb_NOPIECE;			/* it marks the end of list */

	bs[0] = tb_E5;
	bs[1] = tb_NOSQUARE;		/* it marks the end of list */

	bp[0] = tb_KING;
	bp[1] = tb_NOPIECE;			/* it marks the end of list */

#else

	/* needs 4-pc installed */
	/* FEN: 8/8/6p1/4K3/7b/8/8/2k5 w - - 0 76 */

	stm      = tb_WHITE_TO_MOVE;/* 0 = white to move, 1 = black to move */
	epsquare = tb_NOSQUARE;		/* no ep available */
	castling = tb_NOCASTLE;		/* no castling available, otherwise combine all 
									the castling possibilities with '|', for instance
									white could castle both sides, black can't:	 
									castling = tb_WOO | tb_WOOO; 
									both could castle on the king side:	 
									castling = tb_WOO | tb_WOO;
									etc. 
								*/

	ws[0] = tb_E5;
	ws[1] = tb_NOSQUARE;		/* it marks the end of list */

	wp[0] = tb_KING;
	wp[1] = tb_NOPIECE;			/* it marks the end of list */

	bs[0] = tb_H4;
	bs[1] = tb_G6;
	bs[2] = tb_C1;
	bs[3] = tb_NOSQUARE;		/* it marks the end of list */

	bp[0] = tb_BISHOP;
	bp[1] = tb_PAWN;
	bp[2] = tb_KING;
	bp[3] = tb_NOPIECE;			/* it marks the end of list */

#endif

	/*--------------------------------------*\
	|
	|      	PROBING TBs #1 (HARD)
	|   
	\*--------------------------------------*/

	/* 
		probing hard will go to the cache first, if the info is not found there, 
		it will finally go to the Hard Drive to find it
	*/		

	tb_available = tb_probe_hard (stm, epsquare, castling, ws, bs, wp, bp, &info, &pliestomate);

	/* print info */
	dtm_print (stm, tb_available, info, pliestomate);

	/*--------------------------------------*\
	|
	|   ASSIGNING POSITIONAL VALUES for
	|   another example
	|   
	\*--------------------------------------*/

	/* only the rook position is different, the rest is the same */
	ws[1] = tb_B6;

	/*--------------------------------------*\
	|
	|      	PROBING TBs #2 (SOFT)
	|   
	\*--------------------------------------*/

	/* 
		probing soft goes to cache, if info not found there, it returns FALSE
		It will **NEVER** go to the Hard Drive
		If info is found, it is because the previous probe #1 filled up 
		the cache with the info needed for probe #2
	*/	

	tb_available = tb_probe_soft (stm, epsquare, castling, ws, bs, wp, bp, &info, &pliestomate);

	/* print info */
	dtm_print (stm, tb_available, info, pliestomate);

	/*--------------------------------------*\
	|
	|      	PROBING TBs #3 (SOFT)
	|		An example of what happens 
	|		after tbcache_flush()
	|		which may be used to clear it
	|		for epd tests, etc.
	|   
	\*--------------------------------------*/

	/* 
		cache is flushed, so probing soft with the same position as #2 
		will surely return FALSE 
	*/

	tbcache_flush();

	/* same as #2 */
	tb_available = tb_probe_soft (stm, epsquare, castling, ws, bs, wp, bp, &info, &pliestomate);

	/* print info */
	dtm_print (stm, tb_available, info, pliestomate);

	/*--------------------------------------*\
	|
	|      	PROBING TBs #4 
	|		(HARD, only win, draw, lose)
	|   
	\*--------------------------------------*/

	/* 
		Probing with the WDL versions of the probing functions
		will return only the info needed to know whether a position
		is a win, draw, or a loss.  
		The Gaviota tablebase library will try to return this info
		with the best performance possible. If the only info needed for
		a position is WDL, this function should be used rather
		than the regular tb_probe_hard() function.
		This function would be the "equivalent" of one that probes a bitbase.
	*/

	tb_available = tb_probe_WDL_hard (stm, epsquare, castling, ws, bs, wp, bp, &info);

	/* print info */
	wdl_print (stm, tb_available, info);


	/*--------------------------------------*\
	|
	|      	RESTART?
	|		What if the user changes 
	|		the conditions during run?
	|   
	\*--------------------------------------*/

	/* 
	|	NEW INFO BY THE USER, example
	\*---------------------------------------------*/	
	scheme = tb_CP2; /* compression scheme changes */
	path1 = "gtb/gtb2"; 
	path2 = "gtb/gtb1";
	cache_size = 16*1024*1024; /* 16 MiB is the new cache size */
	wdl_fraction = 104; /* more cache for wdl info than before */ 

	/* 
	|	RESTART PROCESS
	\*---------------------------------------------*/	

	/* cleanup old paths */
	paths = tbpaths_done(paths); 

	/* init new paths */
	paths = tbpaths_init(); 
	paths = tbpaths_add (paths, path1);
	paths = tbpaths_add (paths, path2);

	/* restart */
	initinfo = tb_restart (verbosity, scheme, paths);
	tbcache_restart(cache_size, wdl_fraction); 

	/* information to be output for the user, or to be saved in logs etc.*/
	if (initinfo != NULL)
		printf ("%s",initinfo);

	/* 
	|	Just to show, It is possible to know what TBs are installed. 
	|	But, I can only find out after tb_init or tb_restart
	\*----------------------------------------------------------------------------------------*/	

	{
		unsigned av = tb_availability();

		if (0 != (av& 1)) printf ("Some 3-pc TBs available\n"); else printf ("No 3-pc TBs available\n");
		if (0 != (av& 2)) printf ("3-pc TBs complete\n");  
		if (0 != (av& 4)) printf ("Some 4-pc TBs available\n"); else printf ("No 4-pc TBs available\n");
		if (0 != (av& 8)) printf ("4-pc TBs complete\n");  
		if (0 != (av&16)) printf ("Some 5-pc TBs available\n"); else printf ("No 5-pc TBs available\n");
		if (0 != (av&32)) printf ("5-pc TBs complete\n");  
		printf ("\n");
	}

	/* 
	|	Now that TBs have been restarted, we probe once again (HARD) 
	\*----------------------------------------------------------------------------------------*/		
	tb_available = tb_probe_hard (stm, epsquare, castling, ws, bs, wp, bp, &info, &pliestomate);

	/* print info */
	dtm_print (stm, tb_available, info, pliestomate);


	/*--------------------------------------*\
	|
	|	Clean up at the end of the program
	|
	\*--------------------------------------*/

	tbcache_done();

	tb_done();

	paths = tbpaths_done(paths);

	/*--------------------------------------*\
	|
	|         		Return
	|
	\*--------------------------------------*/

	if (tb_available)
		return EXIT_SUCCESS;
	else
		return EXIT_FAILURE;
} 


/*----------------------------------------------------------------------*\
|	These are local functions that just print the results after probing
\*----------------------------------------------------------------------*/

static void
dtm_print (unsigned stm, int tb_available, unsigned info, unsigned pliestomate)
{
	if (tb_available) {

		if (info == tb_DRAW)
			printf ("Draw\n");
		else if (info == tb_WMATE && stm == tb_WHITE_TO_MOVE)
			printf ("White mates, plies=%u\n", pliestomate);
		else if (info == tb_BMATE && stm == tb_BLACK_TO_MOVE)
			printf ("Black mates, plies=%u\n", pliestomate);
		else if (info == tb_WMATE && stm == tb_BLACK_TO_MOVE)
			printf ("Black is mated, plies=%u\n", pliestomate);
		else if (info == tb_BMATE && stm == tb_WHITE_TO_MOVE)
			printf ("White is mated, plies=%u\n", pliestomate);         
		else {
			printf ("FATAL ERROR, This should never be reached\n");
			exit(EXIT_FAILURE);
		}
		printf ("\n");
	} else {
		printf ("Tablebase info not available\n\n");   
	}
}

static void
wdl_print (unsigned stm, int tb_available, unsigned info)
{
	if (tb_available) {

		if (info == tb_DRAW)
			printf ("Draw\n");
		else if (info == tb_WMATE && stm == tb_WHITE_TO_MOVE)
			printf ("White mates\n");
		else if (info == tb_BMATE && stm == tb_BLACK_TO_MOVE)
			printf ("Black mates\n");
		else if (info == tb_WMATE && stm == tb_BLACK_TO_MOVE)
			printf ("Black is mated\n");
		else if (info == tb_BMATE && stm == tb_WHITE_TO_MOVE)
			printf ("White is mated\n");         
		else {
			printf ("FATAL ERROR, This should never be reached\n");
			exit(EXIT_FAILURE);
		}
		printf ("\n");
	} else {
		printf ("Tablebase info not available\n\n");   
	}
}



